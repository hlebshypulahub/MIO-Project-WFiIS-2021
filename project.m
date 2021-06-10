warning off;
clear;
clc;
close all;

%%--%%--%% Zmienne pomocnicze
perc = 0.8;
% round_range = 0.3;
bound_perc = 0.1;
it = 30;

Description = strings(it, 1);
SC_u = zeros(it, 1);
PSO_u = zeros(it, 1);
SC_t = zeros(it, 1);
PSO_t = zeros(it, 1);



%%--%%--%% Inicjalizacja zbioru (iris)
% [dataset, value] = seeds_dataset;
% dataset = dataset.';
% value = vec2ind(value)';
% dataset = [dataset, value];
% [n, ~] = size(dataset);

%%--%%--%% Inicjalizacja zbioru (seeds)
dataset = readmatrix('seeds.csv');
[n, ~] = size(dataset);

%%--%%--%% Pętla na uruchomienie obu algorytmów n razy
for loop = 1:it

dataset = dataset(randperm(size(dataset, 1)), :);

x_u = dataset(1:n*perc, 1:end-1);
setGlobal_x_u(x_u);
y_u = dataset(1:n*perc, end);
setGlobal_y_u(y_u);

x_t = dataset(n*perc+1:end, 1:end-1);
setGlobal_x_t(x_t);
y_t = dataset(n*perc+1:end, end);
setGlobal_y_t(y_t);

%%--%%--%% Inicjalizacja FIS

% showrule(getGlobalfis);

% fuzzy(getGlobalfis);


%%--%%--%% Wykresy funkcji przynależności na każdym wejściu
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

    
    fprintf('Iteracja: %d\n', loop);

    Description(loop) = convertCharsToStrings(sprintf('Iteracja: %d', loop));
    
    optionsSC = genfisOptions('SubtractiveClustering');
    fis = genfis(x_u, y_u, optionsSC);
    setGlobalfis(fis);
    
%     fuzzy(fis);

    %%--%%--%% Ewalujemy FIS
    y_out = evalfis(getGlobalfis, x_u);
    y_test = evalfis(getGlobalfis, x_t);


    %%--%%--%% WYpisywanie liczby dobrze zkwalifikowanych przypadków
    %%--%%--%% set uczący (SubtractiveClustering FIS)
    y_temp = y_out;
    for i = 1:size(y_temp, 1)
        %     if y_temp(i) >= round(y_temp(i)) - round_range && y_temp(i) <= round(y_temp(i)) + round_range
        y_temp(i) = round(y_temp(i));
        %     end
    end
    temp = y_temp - getGlobal_y_u;
    q = find(temp == 0);
    fprintf('Liczba dobrze zkwalifikowanych przypadków (SubtractiveClustering FIS) - set uczący: %d\n', size(q, 1));
    SC_u(loop) = size(q, 1);


    %%--%%--%% WYpisywanie liczby dobrze zkwalifikowanych przypadków
    %%--%%--%% set testujący (SubtractiveClustering FIS)
    y_temp = y_test;
    for i = 1:size(y_temp, 1)
        %     if y_temp(i) >= round(y_temp(i)) - round_range && y_temp(i) <= round(y_temp(i)) + round_range
        y_temp(i) = round(y_temp(i));
        %     end
    end
    temp = y_temp - getGlobal_y_t;
    q = find(temp == 0);
    fprintf('Liczba dobrze zkwalifikowanych przypadków (SubtractiveClustering FIS) - set testujący: %d\n', size(q, 1));
    SC_t(loop) = size(q, 1);


    %%--%%--%% Wykresy wyników (SubtractiveClustering FIS)
    % figure;
    % subplot(2, 1, 1)
    % scatter(1:n*perc, y_out, 55, 'r', 'd')
    % hold on;
    % scatter(1:n*perc, y_u, 'b', 'filled')
    % legend('ymodel', 'yreal')
    % title('Zbior uczacy');
    % subplot(2, 1, 2)
    % scatter(1:(n - n * perc), y_test, 55, 'r', 'd')
    % hold on;
    % scatter(1:(n - n * perc), y_t, 'b', 'filled')
    % legend('ymodel', 'yreal')
    % title('Zbior testujacy');


    %%--%%--%% Pobranie parametrów FIS
    [in, out] = getTunableSettings(getGlobalfis);
    paramVals = getTunableValues(getGlobalfis, [in; out]);


    %%--%%--%% Definiowanie granic na potrzeby PSO
    lb = [];
    ub = [];
    bound_in = [];

    for i = 1:size(dataset, 2) - 1
        temp = fis.Inputs(i).MembershipFunctions.Parameters;
        bound_in = [bound_in, size(fis.Inputs(i).MembershipFunctions, 2) * size(temp, 2)];
    end

    for r = 1:size(dataset, 2) - 1
        for i = 1:bound_in
            lb(end+1) = fis.Inputs(r).Range(1) - fis.Inputs(r).Range(1) * bound_perc;
            ub(end+1) = fis.Inputs(r).Range(2) + fis.Inputs(r).Range(2) * bound_perc;
        end
    end

    temp = fis.Outputs.MembershipFunctions.Parameters;
    bound_out = size(fis.Outputs.MembershipFunctions, 2) * size(temp, 2);

    % for i = 1:bound_out
    %     lb = [lb, fis.Outputs.Range(1) - 0.5];
    %     ub = [ub, fis.Outputs.Range(2) + 0.5];
    % end


    %%--%%--%% Wywołanie PSO
    close all;
    %%% 'SwarmSize', 20, 'MaxIterations', 1000*(size(paramVals, 2) - bound_out)/20, 'MaxStallIterations', 30
    options = optimoptions('particleswarm', 'PlotFcns', @pswplotbestf, ...
        'SwarmSize', 20, 'MaxIterations', 1000*(size(paramVals, 2) - bound_out)/20, 'MaxStallIterations', 30);
    x = particleswarm(@fun, size(paramVals, 2)-bound_out, lb, ub, options);


    %%--%%--%% Inicjalizacja FIS na podstawie danych otrzymanych z PSO
    fis = setTunableValues(fis, in, x);
    y_out = evalfis(fis, x_u);
    y_test = evalfis(fis, x_t);

    % fuzzy(fis);


    %%--%%--%% WYpisywanie liczby dobrze zkwalifikowanych przypadków
    %%--%%--%% set uczący (PSO FIS)
    y_temp = y_out;
    for i = 1:size(y_temp, 1)
        %     if y_temp(i) >= round(y_temp(i)) - round_range && y_temp(i) <= round(y_temp(i)) + round_range
        y_temp(i) = round(y_temp(i));
        %     end
    end
    temp = y_temp - getGlobal_y_u;
    q = find(temp == 0);
    fprintf('Liczba dobrze zkwalifikowanych przypadków (PSO FIS) - set uczący: %d\n', size(q, 1));
    PSO_u(loop) = size(q, 1);


    %%--%%--%% WYpisywanie liczby dobrze zkwalifikowanych przypadków
    %%--%%--%% set testujący (PSO FIS)
    y_temp = y_test;
    for i = 1:size(y_temp, 1)
        %     if y_temp(i) >= round(y_temp(i)) - round_range && y_temp(i) <= round(y_temp(i)) + round_range
        y_temp(i) = round(y_temp(i));
        %     end
    end
    temp = y_temp - getGlobal_y_t;
    q = find(temp == 0);
    fprintf('Liczba dobrze zkwalifikowanych przypadków (PSO FIS) - set testujący: %d\n', size(q, 1));
    PSO_t(loop) = size(q, 1);


    %%--%%--%% Wykresy wyników (PSO FIS)
    % figure;
    % subplot(2, 1, 1)
    % scatter(1:n*perc, y_out, 55, 'r', 'd')
    % hold on;
    % scatter(1:n*perc, y_u, 'b', 'filled')
    % legend('ymodel', 'yreal')
    % title('Zbior uczacy');
    % subplot(2, 1, 2)
    % scatter(1:(n - n * perc), y_test, 55, 'r', 'd')
    % hold on;
    % scatter(1:(n - n * perc), y_t, 'b', 'filled')
    % legend('ymodel', 'yreal')
    % title('Zbior testujacy');

    fprintf('\n\n\n\n');
end


Description(end+1) = "Suma";
SC_u(end+1) = sum(SC_u(1:it));
SC_t(end+1) = sum(SC_t(1:it));
PSO_u(end+1) = sum(PSO_u(1:it));
PSO_t(end+1) = sum(PSO_t(1:it));


Description(end+1) = "Średnie";
SC_u(end+1) = mean(SC_u(1:it));
SC_t(end+1) = mean(SC_t(1:it));
PSO_u(end+1) = mean(PSO_u(1:it));
PSO_t(end+1) = mean(PSO_t(1:it));


Description(end+1) = "Odchylenie std";
SC_u(end+1) = std(SC_u(1:it));
SC_t(end+1) = std(SC_t(1:it));
PSO_u(end+1) = std(PSO_u(1:it));
PSO_t(end+1) = std(PSO_t(1:it));


result = table(Description, SC_u, PSO_u, SC_t, PSO_t);
fprintf('Result:\n');
disp(result);


%%--%%--%% Funkcja Fitness
function fitness = fun(x)

% round_range = 0.3;

for i = 1:size(x, 2)
    if x(i) == 0
        x(i) = 0.001 + rand * (0.05 - 0.001);
    end
end

fis_test = getGlobalfis;
in = getTunableSettings(fis_test);

paramVals = x;
fis_test = setTunableValues(fis_test, in, paramVals);

y_pso = evalfis(fis_test, getGlobal_x_u);

for i = 1:size(y_pso, 1)
    %     if y_pso(i) >= round(y_pso(i)) - round_range && y_pso(i) <= round(y_pso(i)) + round_range
    y_pso(i) = round(y_pso(i));
    %     end
end

temp = y_pso - getGlobal_y_u;
q = find(temp == 0);

fitness = (size(y_pso, 1) - size(q, 1)) / size(y_pso, 1);

end


%%--%%--%% Zmienne globalne
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


function setGlobal_x_t(val)
global x_t
x_t = val;
end

function r = getGlobal_x_t
global x_t
r = x_t;
end


function setGlobal_y_t(val)
global y_t
y_t = val;
end

function r = getGlobal_y_t
global y_t
r = y_t;
end