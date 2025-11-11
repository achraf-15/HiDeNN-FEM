addpath('C:/Users/Lenovo/Downloads/duvant_rule');

filename = 'dunavant_triangle_quadrature.csv';
fid = fopen(filename, 'w');
fprintf(fid, 'rule,order,xi,eta,weight\n');

max_rule_index = 20;  % for example, Dunavant provides rules up to 20
for rule = 1:max_rule_index
    order_num = dunavant_order_num(rule);   % use the library function to get correct order
    if order_num < 1
        continue;
    end
    [xy, w] = dunavant_rule(rule, order_num);
    if isempty(xy) || isempty(w)
        fprintf('Rule %d returned no points, skipping\n', rule);
        continue;
    end
    for i = 1:order_num
        xi = xy(1,i);
        eta = xy(2,i);
        weight = w(i);
        fprintf(fid, '%d,%d,%.16f,%.16f,%.16f\n', rule, order_num, xi, eta, weight);
    end
end

fclose(fid);
fprintf('Finished writing %s\n', filename);
