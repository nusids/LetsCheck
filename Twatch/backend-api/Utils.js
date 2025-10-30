const buildReplyStructure = (rootNode, replyChildren) => {
    let replyStructure = [];
    // Stores the current position of the reply tree we are going to build
    // Stored as an array of indexes, representing the index of the subtree containing our target node at each depth
    // Starting with the root node, so first index array is []
    let replySearchQueue = [[]];
    while (replySearchQueue.length > 0) {
        let targetNode = replyStructure;
        let targetNodeId = rootNode.tweet_id;
        let targetNodeIndex = replySearchQueue.shift();
        const currentIndex = [...targetNodeIndex];
        while (targetNodeIndex.length > 1) {
            const nextLevelIndex = targetNodeIndex.shift();
            targetNode = targetNode[nextLevelIndex].reply_structure;
        }
        // Find all direct children of this node
        for (const [idx, reply] of replyChildren.entries() ) {
            let count = 0;
            if (reply.reply_parent_id === targetNodeId) {
                targetNode.push({
                    'id_str': reply.tweet_id,
                    'unix': reply.unix,
                    'reply_structure': []
                })
                replySearchQueue.push([...currentIndex, count]);
                count += 1;
                replyChildren.splice(idx, 1);
            }
        }
    }
    return replyStructure;
}

const getTweetTrends = (tweets) => {
    let startTime = new Date();

    // Make time segments
    const times = tweets.map(item => item.unix);
    let max = times[times.length - 1];
    let min = times[0];
    const segment = Math.max(1, Math.floor((max - min) / 150));
    max += segment;
    min -= segment;
    const result = [];
    let current = min;

    // Number of tweets per time segment
    while (current + segment <= max) {
        const filtered = tweets.filter(tweet => tweet.unix >= current && tweet.unix <= current + segment);
        let count = 0;
        filtered.forEach(t => {
            count += 1;
        })
        result.push({
            x: Math.floor(current + segment / 2),
            y: count,
        });
        current += segment;
    }

    let now = new Date();
    console.log(`trend processed in ${(now - startTime) / 1000} s`);

    return result;
};

/* Binary search in an array of objects sorted on a given field/column, and return the closest index for given value */
const binarySearch = (arr, value, col) => {
    let firstIndex  = 0,
        lastIndex   = arr.length - 1,
        middleIndex = Math.floor((lastIndex + firstIndex)/2);

    while (arr[middleIndex][col] !== value && firstIndex < lastIndex) {
        if (value < arr[middleIndex]) {
            lastIndex = middleIndex - 1;
        } else if (value > arr[middleIndex]) {
            firstIndex = middleIndex + 1;
        }
        middleIndex = Math.floor((lastIndex + firstIndex)/2);
    }
    return middleIndex;
}

module.exports = {
  buildReplyStructure, getTweetTrends, binarySearch
}