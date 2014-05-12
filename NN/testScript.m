data_cancer;
desired = desired';
train_x = input(1:350, :);
train_y = desired(1:350, :);

test_x = input(351:end, :);
test_y = desired(351:end, :);

[~, nIn] = size(train_x);
[~, nOut] = size(train_y);
nn = nnsetup([nIn, 80, nOut]);
nn.dropConnectFraction = 0.5;
%nn.momentum = .8;
nn.learningRate = .1;
%nn.scaling_learningRate = .999;
opts.numepochs = 1000;
opts.batchsize = 25;
opts.plot = 0;
opts.validation = 1;
[nn, L] = nntrain(nn, train_x, train_y, opts, test_x, test_y);
