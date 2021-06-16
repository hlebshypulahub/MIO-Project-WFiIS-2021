warning off;
clear;
clc;
close all;



%%--%%--%% Zmienne pomocnicze
perc = 0.8;
bound_perc = 0.2;
it = 2;

Description = strings(it, 1);
SC_u = zeros(it, 1);
PSO_u = zeros(it, 1);
SC_t = zeros(it, 1);
PSO_t = zeros(it, 1);


%%--%%--%% Inicjalizacja zbioru (iris)
[dataset, value] = iris_dataset;
dataset = dataset.';
value = vec2ind(value)';
dataset = [dataset, value];
[n, ~] = size(dataset);

%%--%%--%% Inicjalizacja zbioru (seeds)
% dataset = readmatrix('seeds.csv');
% [n, ~] = size(dataset);

%%--%%--%% Inicjalizacja zbioru (wine)
% [dataset, value] = wine_dataset;
% dataset = dataset.';
% value = vec2ind(value)';
% dataset = [dataset(:,1:7), value];
% [n, ~] = size(dataset);

%%--%%--%% Pętla na uruchomienie obu algorytmów n razy
for loop = 1:it

    %%% Randomizacja (mieszanie) datasetu
    dataset = dataset(randperm(size(dataset, 1)), :);

    %%% Inicjalizacja setu uczącego + zmienne globalne
    x_u = dataset(1:n*perc, 1:end-1);
    setGlobal_x_u(x_u);
    y_u = dataset(1:n*perc, end);
    setGlobal_y_u(y_u);

    %%% Inicjalizacja setu testującego + zmienne globalne
    x_t = dataset(n*perc+1:end, 1:end-1);
    setGlobal_x_t(x_t);
    y_t = dataset(n*perc+1:end, end);
    setGlobal_y_t(y_t);


    fprintf('Iteracja: %d\n', loop);
    Description(loop) = convertCharsToStrings(sprintf('Iteracja: %d (%%)', loop));

    %%--%%--%% Inicjalizacja FIS
    optionsSC = genfisOptions('SubtractiveClustering');
    fis = genfis(x_u, y_u, optionsSC);
    setGlobalfis(fis);
    % fuzzy(getGlobalfis);


    %%--%%--%% Ewalujemy FIS
    y_out = evalfis(getGlobalfis, x_u);
    y_test = evalfis(getGlobalfis, x_t);


    %%--%%--%% WYpisywanie liczby dobrze zkwalifikowanych przypadków
    %%--%%--%% set uczący (SubtractiveClustering FIS)
    y_temp = y_out;
    for i = 1:size(y_temp, 1)
        y_temp(i) = round(y_temp(i));
    end
    temp = y_temp - getGlobal_y_u;
    q = find(temp == 0);
    fprintf('Procent dobrze zkwalifikowanych przypadków (SubtractiveClustering FIS) - set uczący: %.3f%%\n', round(size(q, 1) / size(y_out, 1), 5) * 100);
    SC_u(loop) = round(size(q, 1) / size(y_out, 1), 5) * 100;


    %%--%%--%% WYpisywanie liczby dobrze zkwalifikowanych przypadków
    %%--%%--%% set testujący (SubtractiveClustering FIS)
    y_temp = y_test;
    for i = 1:size(y_temp, 1)
        y_temp(i) = round(y_temp(i));
    end
    temp = y_temp - getGlobal_y_t;
    q = find(temp == 0);
    fprintf('Procent dobrze zkwalifikowanych przypadków (SubtractiveClustering FIS) - set testujący: %.3f%%\n', round(size(q, 1) / size(y_test, 1), 5) * 100);
    SC_t(loop) = round(size(q, 1) / size(y_test, 1), 5) * 100;


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


    %%--%%--%% Pobranie parametrów FIS (na potrzeby PSO)
    [in, out] = getTunableSettings(getGlobalfis);
    paramVals = getTunableValues(getGlobalfis, [in; out]);


    %%--%%--%% Definiowanie granic na potrzeby PSO
    lb = [];
    ub = [];
    %%% Ilość parametrów funkcji przynależności na każdym wejściu
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
    

    %%--%%--%% Wywołanie PSO
    close all;
    options = optimoptions('particleswarm', 'PlotFcns', @pswplotbestf, ...
        'SwarmSize', 20, 'MaxIterations', 1000*(sum(bound_in))/20, 'MaxStallIterations', 15);
    x = particleswarm(@fun, sum(bound_in), lb, ub, options);


    %%--%%--%% Inicjalizacja FIS na podstawie danych otrzymanych z PSO
    fis = setTunableValues(fis, in, x);
    y_out = evalfis(fis, x_u);
    y_test = evalfis(fis, x_t);
    % fuzzy(fis);


    %%--%%--%% WYpisywanie liczby dobrze zkwalifikowanych przypadków
    %%--%%--%% set uczący (PSO FIS)
    y_temp = y_out;
    for i = 1:size(y_temp, 1)
        y_temp(i) = round(y_temp(i));
    end
    temp = y_temp - getGlobal_y_u;
    q = find(temp == 0);
    fprintf('Procent dobrze zkwalifikowanych przypadków (PSO FIS) - set uczący: %.3f%%\n', round(size(q, 1) / size(y_out, 1), 5) * 100);
    PSO_u(loop) = round(size(q, 1) / size(y_out, 1), 5) * 100;


    %%--%%--%% WYpisywanie liczby dobrze zkwalifikowanych przypadków
    %%--%%--%% set testujący (PSO FIS)
    y_temp = y_test;
    for i = 1:size(y_temp, 1)
        y_temp(i) = round(y_temp(i));
    end
    temp = y_temp - getGlobal_y_t;
    q = find(temp == 0);
    fprintf('Procent dobrze zkwalifikowanych przypadków (PSO FIS) - set testujący: %.3f%%\n', round(size(q, 1) / size(y_test, 1), 5) * 100);
    PSO_t(loop) = round(size(q, 1) / size(y_test, 1), 5) * 100;


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


Description(end+1) = "Średnie (%)";
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
%%% Funkcje przynależności są typu Gaussa, nie mogą przyjmować 0 jako
%%% parameter odchylenia std, dlatego jeżeli jakaś wartość wektora jest 0,
%%% randomizujemy wartość bliską do 0 i zmieniamy x(i)
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
    y_pso(i) = round(y_pso(i));
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