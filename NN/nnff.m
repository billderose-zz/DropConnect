function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x; %layer activations

    %feedforward pass
    for i = 2 : (n - 1)
        
        %@ The square of the input vector, stored for reuse in sampling
        Vsq = nn.a{i - 1} .*nn.a{i-1}; 
        Wsq = nn.W{i-1}.*nn.W{i-1}; %@
        
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                if (nn.dropConnectFraction > 0)
                    if (~nn.testing)
                        nn.a{i} = sigm(nn.a{i - 1} * (nn.W{i - 1} .* nn.dropConnectMask{i - 1})');
                    else
                        %@ Take samples from a gaussian and average the
                        %output
                        nn.a{i} = sigm(nn.a{i - 1} * (nn.W{i - 1})');
                        nn.a{i} = 0;
                        dropVariance = nn.dropConnectFraction*(1-nn.dropConnectFraction)*Vsq*(Wsq');
                        dropMean = nn.dropConnectFraction*nn.a{i-1}*(nn.W{i-1}');
                        dropStdev = sqrt(dropVariance);
                        
                        % Use the mean by default to speed up
                        % training/validation
                        if (nn.dcTestSamples == 0)
                            nn.a{i} = sigm(dropMean);                                             
                        else
                            for z = 1 : nn.dcTestSamples
                                nn.a{i} = nn.a{i} + sigm(normrnd(dropMean, dropStdev));
                            end
                            nn.a{i} = nn.a{i}.*(1/nn.dcTestSamples);
                        end
                    end
                else
                    nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
                end
            case 'tanh_opt'
                if (nn.dropConnectFraction>0)
                    if (~nn.testing)
                        nn.a{i} = tanh_opt(nn.a{i - 1} * (nn.W{i - 1} .* nn.dropConnectMask{i - 1})');
                    else
                        %@ Take samples from a gaussian and average the
                        %output
                        nn.a{i} = tanh_opt(nn.a{i - 1} * (nn.W{i - 1})');
                        nn.a{i} = 0;
                        dropVariance = nn.dropConnectFraction*(1-nn.dropConnectFraction)*Vsq*(Wsq');
                        dropMean = nn.dropConnectFraction*nn.a{i-1}*(nn.W{i-1}');
                        dropStdev = sqrt(dropVariance);
                        
                        % Use the mean by default to speed up
                        % training/validation
                        if (nn.dcTestSamples == 0)
                            nn.a{i} = tanh_opt(dropMean);
                                             
                        else
                            for z = 1 : nn.dcTestSamples
                                nn.a{i} = nn.a{i} + tanh_opt(normrnd(dropMean, dropStdev));
                            end
                            nn.a{i} = nn.a{i}.*(1/nn.dcTestSamples);
                        end
                    end
                else
                    nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
                end

        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end