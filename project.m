[dataset, value] = iris_dataset;

dataset = dataset.';
value = vec2ind(value)';

dataset = [dataset value];
dataset = dataset(randperm(size(dataset, 1)), :);


[n, ~] = size(dataset);
perc = 0.8;


x_u = dataset(1:n*perc, 1:end-1);
y_u = dataset(1:n*perc, end);

x_t = dataset(n*perc+1:end, 1:end-1);
y_t = dataset(n*perc+1:end, end);


options = genfisOptions('SubtractiveClustering');
fis = genfis(x_u, y_u, options);

showrule(fis);

fuzzy(fis);


figure;
[x,mf] = plotmf(fis,'input',1);
subplot(4,1,1)
plot(x,mf)
xlabel('Membership Functions for Input 1')
[x,mf] = plotmf(fis,'input',2);
subplot(4,1,2)
plot(x,mf)
xlabel('Membership Functions for Input 2')
[x,mf] = plotmf(fis,'input',3);
subplot(4,1,3)
plot(x,mf)
xlabel('Membership Functions for Input 3')
[x,mf] = plotmf(fis,'input',4);
subplot(4,1,4)
plot(x,mf)
xlabel('Membership Functions for Input 4')


y_out = evalfis(fis, x_u);
y_test = evalfis(fis, x_t);



figure;
subplot(2,1,1)
plot(1:n*perc, y_out, 'r')
hold on;
plot(1:n*perc, y_u, 'b')
legend('ymodel', 'yreal')
title('Zbior uczacy');


subplot(2,1,2)
plot(1:(n-n*perc), y_test, 'r')
hold on;
plot(1:(n-n*perc), y_t, 'b')
legend('ymodel', 'yreal')
title('Zbior testujacy');













