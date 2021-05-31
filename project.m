warning off;
clear;
clc;
close all;

[dataset, value] = iris_dataset;

dataset = dataset.';
value = vec2ind(value)';


dataset = [dataset, value];
dataset = dataset(randperm(size(dataset, 1)), :);


[n, ~] = size(dataset);
perc = 0.8;


x_u = dataset(1:n*perc, 1:end-1);
setGlobal_x_u(x_u);
y_u = dataset(1:n*perc, end);
setGlobal_y_u(y_u);

x_t = dataset(n*perc+1:end, 1:end-1);
y_t = dataset(n*perc+1:end, end);


options = genfisOptions('SubtractiveClustering');
fis = genfis(x_u, y_u, options);


setGlobalfis(fis);


showrule(getGlobalfis);

fuzzy(getGlobalfis);


% figure;
% [x,mf] = plotmf(fis,'input',1);
% subplot(4,1,1)
% plot(x,mf)
% xlabel('Membership Functions for Input 1')
% [x,mf] = plotmf(fis,'input',2);
% subplot(4,1,2)
% plot(x,mf)
% xlabel('Membership Functions for Input 2')
% [x,mf] = plotmf(fis,'input',3);
% subplot(4,1,3)
% plot(x,mf)
% xlabel('Membership Functions for Input 3')
% [x,mf] = plotmf(fis,'input',4);
% subplot(4,1,4)
% plot(x,mf)
% xlabel('Membership Functions for Input 4')


y_out = evalfis(getGlobalfis, x_u);
y_test = evalfis(getGlobalfis, x_t);


figure;
subplot(2,1,1)
scatter(1:n*perc, y_out, 'r','filled')
hold on;
scatter(1:n*perc, y_u, 'b','filled')
legend('ymodel', 'yreal')
title('Zbior uczacy');


subplot(2,1,2)
scatter(1:(n-n*perc), y_test, 'r','filled')
hold on;
scatter(1:(n-n*perc), y_t, 'b','filled')
legend('ymodel', 'yreal')
title('Zbior testujacy');


[in, out] = getTunableSettings(getGlobalfis);
paramVals = getTunableValues(getGlobalfis, [in; out]);
lb = [];
ub = [];
% lb = zeros(size(paramVals));
% ub = ones(size(paramVals));

bound_in = [];
for i = 1:size(dataset, 2) - 1
    temp = fis.Inputs(i).MembershipFunctions.Parameters;
    bound_in = [bound_in, size(fis.Inputs(i).MembershipFunctions, 2) * size(temp, 2)];
end

for r = 1:size(dataset, 2) - 1
    fis.Inputs(r).Range(1)
    for i = 1:bound_in
        lb(end+1) = fis.Inputs(r).Range(1) - fis.Inputs(r).Range(1) * 0.2;
        ub(end+1) = fis.Inputs(r).Range(2) + fis.Inputs(r).Range(2) * 0.2;
    end
end

temp = fis.Outputs.MembershipFunctions.Parameters;
bound_out = size(fis.Outputs.MembershipFunctions, 2) * size(temp, 2);

for i = 1:bound_out
    lb = [lb, fis.Outputs.Range(1) - 0.5];
    ub = [ub, fis.Outputs.Range(2) + 0.5];
end

options = optimoptions('particleswarm','PlotFcns',@pswplotbestf,'MaxIter',20,'SwarmSize',20);
x = particleswarm(@fun,size(paramVals,2),lb,ub,options);


fis = setTunableValues(fis,[in;out],x);
y_out = evalfis(fis, x_u);
y_test = evalfis(fis, x_t);
fuzzy(fis);
figure;
subplot(2,1,1)
scatter(1:n*perc, y_out, 'r','filled')
hold on;
scatter(1:n*perc, y_u, 'b','filled')
legend('ymodel', 'yreal')
title('Zbior uczacy');

subplot(2,1,2)
scatter(1:(n-n*perc), y_test, 'r','filled')
hold on;
scatter(1:(n-n*perc), y_t, 'b','filled')
legend('ymodel', 'yreal')
title('Zbior testujacy');




function fitness = fun(x)
    
for i = 1:size(x,2)
    if x(i)==0
        x(i) = 0.001 + rand*(0.05-0.001);
    end
end

    fis_test = getGlobalfis;
    [in,out] = getTunableSettings(fis_test);

    paramVals = getTunableValues(fis_test,[in;out]);

    paramVals = x;
    fis_test = setTunableValues(fis_test,[in;out],paramVals);

%     fuzzy(fis_test);

    y_pso = evalfis(fis_test, getGlobal_x_u);

    y_pso = round(y_pso);

    temp = y_pso - getGlobal_y_u;
    q = find(temp);

    fitness = size(q) / size(y_pso);

end
%
%
function setGlobalfis(val)
global fis
fis = val;
end

function r = getGlobalfis
global fis
r = fis;
end


function setGlobal_x_u(val)
global x_u
x_u = val;
end

function r = getGlobal_x_u
global x_u
r = x_u;
end


function setGlobal_y_u(val)
global y_u
y_u = val;
end

function r = getGlobal_y_u
global y_u
r = y_u;
end
