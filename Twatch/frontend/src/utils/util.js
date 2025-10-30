export const getMax = (arr) => {
  let max = -Infinity;

  for (let i = 0; i < arr.length; i += 1) {
    max = arr[i] > max ? arr[i] : max;
  }

  return max;
};

export const getMin = (arr) => {
  let min = Infinity;

  for (let i = 0; i < arr.length; i += 1) {
    min = arr[i] < min ? arr[i] : min;
  }

  return min;
};

export const getTweetTrends = (times) => {
  let max = getMax(times);
  let min = getMin(times);

  if (min === max) {
    return [{
      x: max,
      y: times.length,
    }];
  }

  const segment = Math.floor((max - min) / 150);
  max += segment;
  min -= segment;
  const result = [];
  let current = min;


  while (current + segment <= max) {
    const filtered = times.filter(item => item >= current && item <= current + segment);
    result.push({
      x: Math.floor(current + segment / 2),
      y: filtered.length,
    });
    current += segment;
  }

  return result;
};

export const toHttps = (url) => {
  if (url.match('^http:')) {
    return `https${url.slice(4)}`;
  }
  return url;
};
