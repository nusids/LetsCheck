const express = require("express");
const router = express.Router();
const {
  SearchUtils
} = require('./searchUtils');
const Utils = require('./Utils')
const s = new SearchUtils();


router.get("/", function (req, res) {
  res.json({
    status: "API Working"
  });
});

// API for storing queries, which will be handled by backend
router.post("/get_pending_requests", async (req, res) => {
  res.json(s.getPendingRequests());
})

router.post('/remove_pending_request', async (req, res) => {
  const reqId = req.query.request_id;
  res.json(s.removePendingRequest(reqId));
});

router.post('/save_pending_result', async (req, res) => {
  const reqId = req.body.request_id;
  const re = req.body.result;
  s.saveResult(reqId, re);
  res.sendStatus(200);
})

// API for sending a custom request
router.post("/custom", async (req, res) => {
  try {
    let result = await s.fetchCustomRequest(req.body);
    await res.json(result);
  } catch (e) {
    console.log(e);
    res.sendStatus(404);
  }
})

// Get tweets of a specific event with tweet ids, separated by comma
// LetsCheck: since we no longer have events, this will be the same as fetchBatchTweets
router.get("/tweets", async (req, res) => {
  let tweetIds = req.query.id.split(",");
  try {
    let result = await s.fetchTweetsById(tweetIds);
    await res.json(result);
  } catch (e) {
    console.log(e);
    res.sendStatus(404);
  }
});

router.post("/diffusionData", async (req, res) => {
  if (!req.body || req.body.id.length === 0) {
    res.sendStatus(404);
  }
  console.log("Building diffusion network...")
  const rootTweetIds = req.body.id;
  let result = [];
  for (const rootTweetId of rootTweetIds) {
    let replyChildTweets = [];
    try {
      replyChildTweets = await s.fetchReplyTweetsByRootTweetId([rootTweetId]);
    } catch (e) {
      console.log(e);
      res.sendStatus(404);
    }

    if (replyChildTweets.length === 0) {
      continue
    }

    let rootTweet = {}

    replyChildTweets.forEach((child, i) => {
      if (child.reply_parent_id === "-1") {
        // Is the root tweet itself
        rootTweet = child;
        replyChildTweets.splice(i, 1);
      }
    })
    const replyStructure = Utils.buildReplyStructure(rootTweet,[...replyChildTweets]);
    replyChildTweets = replyChildTweets.map(item => ({
      ...item,
      sentiment: {
        responsetype_vs_source: "comment",
      }
    }));
    result.push({
      ...rootTweet,
      'annotation': {
        misinformation: 0,
        links: [],
        is_turnaround: 0,
        proven_true: 0
      },
      'replies': replyChildTweets,
      'reply_structure': replyStructure
    });
  }
  console.log(`Found ${result.length} threads in total.`);
  res.json(result);
  console.log("Finished building diffusion network...");
  console.log("--------------------------------------")
});

router.get("/overallTrend", async (req, res) => {
  console.log("Getting overall trend...")
  const trendRaw = await s.fetchOverallTrend();
  console.log("----------------------------")
  res.json(trendRaw);
})

router.get("/search", async (req, res) => {

  let queryRaw = req.query.q;

  if (!queryRaw || queryRaw.length === 0) {
    res.sendStatus(404);
  }

  try {
    let result = await s.searchTweets(queryRaw);
    if (result.isValid) {
      res.json(result);
    } else {
      res.sendStatus(404);
    }

  } catch (e) {
    res.sendStatus(500);
    throw e;
  }
});

module.exports = router;
