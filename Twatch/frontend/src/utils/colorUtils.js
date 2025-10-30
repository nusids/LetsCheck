export const hexToRgb = (hex) => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [
    parseInt(result[1], 16),
    parseInt(result[2], 16),
    parseInt(result[3], 16),
  ] : null;
};

export const rgbToHex = (r, g, b) => `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;

// colorChannelA and colorChannelB are ints ranging from 0 to 255
export const colorChannelMixer = (colorChannelA, colorChannelB, amountToMix) => {
  const channelA = colorChannelA * amountToMix;
  const channelB = colorChannelB * (1 - amountToMix);
  return parseInt(channelA + channelB);
};

// rgbA and rgbB are arrays, amountToMix ranges from 0.0 to 1.0
// example (red): rgbA = [255,0,0]
export const colorMixer = (rgbA, rgbB, amountToMix) => {
  const r = colorChannelMixer(rgbA[0], rgbB[0], amountToMix);
  const g = colorChannelMixer(rgbA[1], rgbB[1], amountToMix);
  const b = colorChannelMixer(rgbA[2], rgbB[2], amountToMix);

  return {
    r, g, b,
  };
};
