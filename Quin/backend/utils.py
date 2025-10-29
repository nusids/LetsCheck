from html.parser import HTMLParser


class WebParser(HTMLParser):
    """
    A class for converting the tagged html to formats that can be used by a ML model
    """
    def __init__(self):
        super().__init__()
        self.block_tags = {
            'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
        }
        self.inline_tags = {
            '', 'a', 'b', 'main', 'span', 'em', 'strong', 'br'
        }
        self.allowed_tags = {'div', 'p', '', 'a', 'b', 'main', 'span', 'em', 'strong', 'br'}

        self.opened_tags = []
        self.block_content = ''
        self.blocks = []

    def get_last_opened_tag(self):
        """
        Gets the last visited tag
        :return:
        """
        if len(self.opened_tags) > 0:
            return self.opened_tags[len(self.opened_tags) - 1]
        return ''

    def error(self, message):
        pass

    def handle_starttag(self, tag, attrs):
        """
        Handles the start tag of an HTML node in the tree
        :param tag: the HTML tag
        :param attrs: the tag attributes
        :return:
        """
        self.opened_tags.append(tag)

    def handle_endtag(self, tag):
        """
        Handles the end tag of an HTML node in the tree
        :param tag: the HTML tag
        :return:
        """
        if tag in self.block_tags:
            self.block_content = self.block_content.strip()
            if len(self.block_content) > 0:
                #if not self.block_content.endswith('.'): self.block_content += '.'
                self.blocks.append(self.block_content)
            self.block_content = ''
        if len(self.opened_tags) > 0:
            self.opened_tags.pop()

    def handle_data(self, data):
        """
        Handles a text HTML node in the tree
        :param data: the text node
        :return:
        """
        last_opened_tag = self.get_last_opened_tag()
        if last_opened_tag in self.allowed_tags:
            data = data.replace('  ', ' ').strip()
            if data != '':
                self.block_content += data + ' '

    def get_blocks(self):
        return self.blocks
