function [ out ] = sigm( x )
%SIGM Summary of this function goes here
%   Detailed explanation goes here
out = 1 ./ (1 + exp(-x));
end

