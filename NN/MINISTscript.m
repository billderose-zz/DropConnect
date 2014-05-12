%dat = csvread('MINIST.csv', 1, 0);
%desired = dat(:,1);
%input = dat(:,2:end);
%dHot = zeros(size(desired,1),10);
for k = 1:size(desired,1)
    dHot(k,desired(k)+1) = 1;
end
train_x = input(1:16000, :);
train_y = dHot(1:16000, :);

test_x = input(16001:19000, :);
test_y = dHot(16001:19000, :);

[~, nIn] = size(train_x);
[~, nOut] = size(train_y);
nn = nnsetup([nIn, 400, nOut]);
nn.dropConnectFraction = 0.5;
nn.output = 'softmax';
nn.momentum = .8;
nn.learningRate = .01;
nn.scaling_learningRate = .999;
opts.numepochs = 500;
opts.batchsize = 200;
opts.plot = 1;
opts.validation = 1;
[nn, L] = nntrain(nn, train_x, train_y, opts, test_x, test_y);