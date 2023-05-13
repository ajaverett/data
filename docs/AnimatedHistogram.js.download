"use strict";

/**
 * @constructor
 * @param {Element} $div The div to build the histogram in.  Width and height are inherited from this div
 * @param {string} title The title.
 * @param {HistogramData} histogramData The histogram data.
 * @param {boolean} yTicks Display tick marks on the y axis.
 * @param {number} yIntervals The number of intervals to show on the y Axis.
 * @param {boolean} plotStats
 * @param {boolean} editable Is this histogram's data editable by clicking with the mouse.
 */
var AnimatedHistogram = function($div, title, histogramData, yTicks, yIntervals, plotStats, editable) {
    Histogram.call(this, $div, title, histogramData, yTicks, yIntervals, plotStats, editable);
    this.animatedBars = [];
    this.animating = false;
    return this;
};


AnimatedHistogram.prototype = Object.create(Histogram.prototype);

/**
 * Animate in the next data point
 */
AnimatedHistogram.prototype.animateNextDataPoint = function()
{
  if (this.samples.length === 0) {
    this.animating = false;
    this.setHistogramData(this.histogramData); // Setting data using this accessor causes display to be updated
    // Remove animated bar elements
    this.animatedBars.forEach(function($bar) {
      $bar.remove();
    });
    this.animatedBars = [];
    this.notifyObserversOfChange();
    if (this.animationCompletionCallback) {
      this.animationCompletionCallback();
      this.animationCompletionCallback = undefined;
    }
    return;
  }
  var self = this;
  var height = this.frequencyToHeight(1);
  var totalHeight = this.$histogramContentArea.height();
  var dataPoint = this.samples.pop();
  var columnIndex = HistogramData.binIndexForDataPoint(dataPoint, self.histogramData.values);
  var frequency = this.histogramData.frequencies[columnIndex];
  this.histogramData.frequencies[columnIndex]++; // Add this data point to our histogram
  var endHeight = this.frequencyToHeight(frequency);
  var $column = this.bars[columnIndex].parent();
  var $animatedBar = $('<div class="histogram_bar"></div>');
  $animatedBar.css('height', height);
  $animatedBar.css('bottom', totalHeight - height);
  $animatedBar.css('background-color', self.barColor);
  $column.append($animatedBar);
  this.animatedBars.push($animatedBar);
  $animatedBar.animate({ 'bottom' : endHeight }, function() {
    // animation complete
    self.animateNextDataPoint();
  });
};


/**
 * Animate in new data.
 * @param {Array<number>} samples An array of data points.
 * @param callback A callback which will be called after all animations are complete.
 */
AnimatedHistogram.prototype.animateInData = function(samples, callback)
{
  if (!this.animating) {
     this.samples = samples.slice(0); // Clone samples array
     this.animateNextDataPoint();
  }
  else {
    this.samples = this.samples.concat(samples);
  }
  this.animationCompletionCallback = callback;
};