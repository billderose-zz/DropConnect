learning_rate = 0.1;
limit = 100;
data_1d_separable    ; train
data_slides          ; train
data_and             ; train
data_or              ; train
data_implies         ; train
data_nand            ; train
data_nor             ; train
data_majority        ; train

limit = 1000;
data_xor             ; train
data_parity          ; train
data_iris_setosa     ; train
data_iris_versicolor ; train
data_iris_virginica  ; train

limit = 10000;
learning_rate = 0.01;
data_cancer          ; train

limit = 100000;
learning_rate = 0.1;
data_housing         ; train

