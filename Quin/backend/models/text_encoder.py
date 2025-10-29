import importlib
import json
import logging
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "11,12,14,15"

import hashlib
import pickle
import torch
torch.cuda.current_device()
import transformers
from torch.optim import AdamW

import numpy as np

from pprint import pprint
from typing import Iterable, Tuple, Type, Dict, List, Union, Optional
from numpy.core.multiarray import ndarray
from torch import nn
from torch.nn import DataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from models.data_utils import get_qnli_examples, get_single_examples, get_ict_examples, get_examples, get_qar_examples, \
    get_qar_artificial_examples, get_retrieval_examples
from tqdm import tqdm

from collections import OrderedDict
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel

from models.vector_index import VectorIndex


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)


def batch_to_device(batch, target_device: torch.device):
    """
    send a batch to a device

    :param batch:
    :param target_device:
    :return: the batch sent to the device
    """
    features = batch['features']
    for paired_sentence_idx in range(len(features)):
        for feature_name in features[paired_sentence_idx]:
            features[paired_sentence_idx][feature_name] = features[paired_sentence_idx][feature_name].to(target_device)

    labels = batch['labels'].to(target_device)
    return features, labels


class BERT(nn.Module):
    """BERT model to generate token embeddings.
    Each token is mapped to an output vector from BERT.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, do_lower_case: Optional[bool] = None, model_args: Dict = {}, tokenizer_args: Dict = {}):
        super(BERT, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        if max_seq_length > 510:
            logging.warning("BERT only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510")
            max_seq_length = 510
        self.max_seq_length = max_seq_length

        if self.do_lower_case is not None:
            tokenizer_args['do_lower_case'] = do_lower_case

        self.model = AutoModel.from_pretrained(model_name_or_path, **model_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        output_states = self.model(**features)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        if len(output_states) > 2:
            features.update({'all_layer_embeddings': output_states[2]})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask
        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 2  ##Add Space for CLS + SEP token

        return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, pad_to_max_length=True, return_tensors='pt')

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
            config = json.load(fIn)
        return BERT(model_name_or_path=input_path, **config)


class SentenceTransformer(nn.Sequential):
    def __init__(self, model_path: str = None, modules: Iterable[nn.Module] = None, device: str = None,
                 parallel=False):
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        if model_path is not None:
            logging.info("Load SentenceTransformer from folder: {}".format(model_path))

            with open(os.path.join(model_path, 'modules.json')) as fIn:
                contained_modules = json.load(fIn)

            modules = OrderedDict()
            for module_config in contained_modules:
                module_class = import_from_string(module_config['type'])
                module = module_class.load(os.path.join(model_path, module_config['path']))
                if 'BERT' in module_config['type']:
                    if parallel:
                        print('Using parallel for SentenceTransformer')
                        module = DataParallel(module)
                modules[module_config['name']] = module

        super().__init__(modules)

        self.best_score = -1
        self.total_steps = 0

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))
        self.device = torch.device(device)
        self.to(device)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            print(score)
            if score > self.best_score and save_best_model:
                print('saving')
                self.save(output_path)
                self.best_score = score

    def evaluate(self, evaluator, output_path: str = None):
        """
        Evaluate the model

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def encode(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None, output_value: str = 'sentence_embedding', convert_to_numpy: bool = True) -> List[ndarray]:
        """
        Computes sentence embeddings
        :param sentences:
           the sentences to embed
        :param batch_size:
           the batch size used for the computation
        :param show_progress_bar:
            Output a progress bar when encode sentences
        :param output_value:
            Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings
            to get wordpiece token embeddings.
        :param convert_to_numpy:
            If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :return:
           Depending on convert_to_numpy, either a list of numpy vectors or a list of pytorch tensors
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel()==logging.INFO or logging.getLogger().getEffectiveLevel()==logging.DEBUG)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                sentence = sentences[idx]
                tokens = self.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = {}
            for text in batch_tokens:
                sentence_features = self.get_sentence_features(text, longest_seq)

                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])

            for feature_name in features:
                try:
                    features[feature_name] = torch.tensor(np.asarray(features[feature_name])).to(self.device)
                except:
                    features[feature_name] = torch.cat(features[feature_name]).to(self.device)

            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                if convert_to_numpy:
                    embeddings = embeddings.to('cpu').numpy()

                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]

        return all_embeddings

    def get_max_seq_length(self):
        if hasattr(self._first_module(), 'max_seq_length'):
            return self._first_module().max_seq_length

        return None

    def tokenize(self, text):
        return self._first_module().tokenize(text)

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self):
        return self._last_module().get_sentence_embedding_dimension()

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        try:
            return self._modules[next(iter(self._modules))].module
        except:
            return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(self, path):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        """
        if path is None:
            return

        logging.info("Save model to {}".format(path))
        contained_modules = []

        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            if isinstance(module, DataParallel):
                module = module.module
            model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            contained_modules.append(
                {'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        with open(os.path.join(path, 'config.json'), 'w') as fOut:
            json.dump({'__version__': '1.0'}, fOut, indent=2)

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0][0])

        labels = []
        paired_texts = [[] for _ in range(num_texts)]
        max_seq_len = [0] * num_texts
        for tokens, label in batch:
            labels.append(label)
            for i in range(num_texts):
                paired_texts[i].append(tokens[i])
                max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

        features = []
        for idx in range(num_texts):
            max_len = max_seq_len[idx]
            feature_lists = {}

            for text in paired_texts[idx]:
                sentence_features = self.get_sentence_features(text, max_len)

                for feature_name in sentence_features:
                    if feature_name not in feature_lists:
                        feature_lists[feature_name] = []

                    feature_lists[feature_name].append(sentence_features[feature_name])

            for feature_name in feature_lists:
                try:
                    feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))
                except:
                    feature_lists[feature_name] = torch.cat(feature_lists[feature_name])

            features.append(feature_lists)

        return {'features': features, 'labels': torch.stack(labels)}

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator,
            epochs: int = 1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            acc_steps: int = 4,
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = 'O1',
            local_rank: int = -1
            ):
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        device = self.device

        for loss_model in loss_models:
            loss_model.to(device)

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = num_train_steps
            if local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                t_total=t_total)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            for train_idx in range(len(loss_models)):
                model, optimizer = amp.initialize(loss_models[train_idx], optimizers[train_idx],
                                                  opt_level=fp16_opt_level)
                loss_models[train_idx] = model
                optimizers[train_idx] = optimizer

        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        num_train_objectives = len(train_objectives)

        for epoch in range(epochs):
            #logging.info('Epoch {}'.format(epoch))
            epoch_loss = 0
            epoch_steps = 0
            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            step_iterator = tqdm(range(steps_per_epoch), desc='loss: -')
            for step in step_iterator:
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        # logging.info("Restart data_iterator")
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = batch_to_device(data, self.device)
                    loss_value = loss_model(features, labels)
                    loss_value = loss_value / acc_steps

                    if fp16:
                        with amp.scale_loss(loss_value, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                    if (step + 1) % acc_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                self.total_steps += 1

                if (step + 1) % acc_steps == 0:
                    epoch_steps += 1
                    epoch_loss += loss_value.item()
                    step_iterator.set_description('loss: {} - acc steps: {}'.format((epoch_loss / epoch_steps),
                                                                                    (self.total_steps / acc_steps)))

                if evaluation_steps > 0 and self.total_steps > 0 and (self.total_steps / acc_steps) % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, epoch_steps)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)

    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))


class SentencesDataset(Dataset):
    def __init__(self, examples: List[InputExample], model: SentenceTransformer, show_progress_bar: bool = None):
        self.examples = examples
        self.model = model

    def __getitem__(self, item):
        tokenized_texts = [model.tokenize(text) for text in self.examples[item].texts]
        return tokenized_texts, torch.tensor(self.examples[item].label, dtype=torch.float)

    def __len__(self):
        return len(self.examples)


class MultipleNegativesRankingLossANN(nn.Module):
    def __init__(self, model: SentenceTransformer, negative_samples=4):
        super(MultipleNegativesRankingLossANN, self).__init__()
        self.model = model
        self.negative_samples = negative_samples

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        return self.multiple_negatives_ranking_loss(embeddings)

    def multiple_negatives_ranking_loss(self, embeddings: List[torch.Tensor]):
        positive_loss = torch.mean(torch.sum(embeddings[0] * embeddings[1], dim=-1))

        negative_loss = torch.sum(embeddings[0] * embeddings[1], dim=-1)
        for i in range(2, self.negative_samples + 2):
            negative_loss = torch.cat((negative_loss, torch.sum(embeddings[0] * embeddings[i], dim=-1)), dim=-1)
        negative_loss = torch.mean(torch.logsumexp(negative_loss, dim=-1))

        return -positive_loss + negative_loss


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        reps_a, reps_b = reps
        return self.multiple_negatives_ranking_loss(reps_a, reps_b)

    def multiple_negatives_ranking_loss(self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor):
        scores = torch.matmul(embeddings_a, embeddings_b.t())
        diagonal_mean = torch.mean(torch.diag(scores))
        mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
        return -diagonal_mean + mean_log_row_sum_exp


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, torch.Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)


class RankingEvaluator:
    def __init__(self, dataloader: DataLoader, random_paragraphs: DataLoader = None,
                 name: str = '', show_progress_bar: bool = None):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.name = name
        if name:
            name = "_" + name

        if show_progress_bar is None:
            show_progress_bar = (
                    logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.random_paragraphs = random_paragraphs

    def __call__(self, model: 'SequentialSentenceEmbedder', output_path: str = None, epoch: int = -1,
                 steps: int = -1) -> float:
        model.eval()
        embeddings1 = []
        embeddings2 = []

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Evaluation" + out_txt)

        logging.info('Calculating sentence embeddings...')

        self.dataloader.collate_fn = model.smart_batching_collate
        iterator = self.dataloader
        evidence_features = []
        for step, batch in enumerate(tqdm(iterator, desc='batch')):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                emb1, emb2 = [model(sent_features)['sentence_embedding'].to("cpu").numpy() for sent_features in
                              features]
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)

            input_ids_2 = features[1]['input_ids'].to("cpu").numpy()
            for f in input_ids_2:
                evidence_features.append(hashlib.sha1(str(f).encode('utf-8')).hexdigest())

        total_examples = len(embeddings1)

        logging.info('Building index...')

        vindex = VectorIndex(len(embeddings1[0]))
        for v in embeddings2:
            vindex.add(v)

        if self.random_paragraphs is not None:
            logging.info('Calculating random wikipedia paragraph embeddings...')

            self.random_paragraphs.collate_fn = model.smart_batching_collate
            iterator = self.random_paragraphs
            for step, batch in enumerate(iterator):
                if step % 10 == 0:
                    logging.info('Batch {}/{}'.format(step, len(iterator)))
                features, label_ids = batch_to_device(batch, self.device)
                with torch.no_grad():
                    embeddings = model(features[0])['sentence_embedding'].to("cpu").numpy()
                for emb in embeddings:
                    vindex.add(emb)

        vindex.build()

        logging.info('Ranking evaluation...')

        mrr = 1e-8
        recall = {1: 0, 5: 0, 10: 0, 20: 0, 100: 0}

        all_results, _ = vindex.search(embeddings1, k=100, probes=1024)

        for i in range(total_examples):
            results = all_results[i]
            rank = 1
            found = False
            for r in results:
                if r < len(evidence_features) and evidence_features[r] == evidence_features[i]:
                    mrr += 1 / rank
                    found = True
                    break
                rank += 1

            for topk, count in recall.items():
                if rank <= topk and found:
                    recall[topk] += 1
        mrr /= total_examples
        for topk, count in recall.items():
            recall[topk] /= total_examples
            logging.info('recall@{} : {}'.format(topk, recall[topk]))

        logging.info('mrr@100 : {}'.format(mrr))

        if output_path is not None:
            f = open(output_path + '/stats.csv', 'a+')
            f.write('{},{},{},{},{},{}\n'.format(recall[1], recall[5], recall[10], recall[20], recall[100], mrr))
            f.close()

        return mrr


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.CRITICAL,
                        handlers=[LoggingHandler()])

    # Read the dataset
    model_name = 'distilroberta-base'
    batch_size = 384
    model_save_path = 'weights/encoder/qrbert-multitask-distil'
    num_epochs = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use BERT for mapping tokens to embeddings
    word_embedding_model = BERT(model_name, max_seq_length=256, do_lower_case=False)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(),
                            pooling_mode_mean_tokens=True,
                            pooling_mode_cls_token=False,
                            pooling_mode_max_tokens=False)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    word_embedding_model = DataParallel(word_embedding_model)
    word_embedding_model.to(device)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], parallel=True)

    #model = SentenceTransformer('models/weights/encoder/sbert-nli-fnli-qar2-768', parallel=True)

    logging.info("Reading dev dataset")
    dev_data = SentencesDataset(get_retrieval_examples(filename='../data/retrieval/dev.jsonl',
                                                       no_statements=False,
                                                       ), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=1024)
    evaluator = RankingEvaluator(dev_dataloader)
    #model.best_score = model.evaluate(evaluator)

    train_data = SentencesDataset(get_retrieval_examples(filename='../data/retrieval/train.jsonl',
                                                         no_statements=False),
                                  model=model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = MultipleNegativesRankingLoss(model=model)

    for i in range(num_epochs):
        logging.info("Epoch {}".format(i))
        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=evaluator,
                  epochs=1,
                  acc_steps=1,
                  evaluation_steps=1000,
                  warmup_steps=1000,
                  output_path=model_save_path,
                  optimizer_params={'lr': 1e-6, 'eps': 1e-6, 'correct_bias': False}
                  )
        torch.cuda.empty_cache()
