'use strict';


/**
 * @constructor
 */
var Stats = function(numberOfObservations, mean, median, sd, range, skew, MAD, variance, kurtosis, varianceu) {
    this.numberOfObservations = numberOfObservations;
    this.mean = mean;
    this.median = median;
    this.sd = sd;
    this.range = range;
    this.skew = skew;
    this.MAD = MAD;
    this.variance = variance;
    this.kurtosis = kurtosis;
    this.varianceu = varianceu;
};


/**
 * @param value A number value
 */
Stats.prototype.formatValue = function(value) {
    if (value === null || value === undefined) {
        return ''
    }
    else {
        return value.toFixed(2);
    }
};
