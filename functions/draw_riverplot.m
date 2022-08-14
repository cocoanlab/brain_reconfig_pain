function draw_riverplot(cont_overlap, net_cols)

figure;
hold on;

ribbon_layers = [];

for img_i = 1:numel(cont_overlap)+1
    
    if img_i == 1
        y_wid = sum(cont_overlap{img_i}, 1)';
    else
        y_wid = sum(cont_overlap{img_i-1}, 2);
    end
    n_rects = numel(y_wid);
    y_wid = y_wid ./ sum(y_wid) * n_rects;
    y_loc = [0; cumsum(y_wid(1:end-1) + 0.1)];
    x_wid = repmat(0.5, n_rects, 1);
    x_loc = repmat(4*(img_i-1), n_rects, 1);

    for i = 1:n_rects
        ribbon_layers{img_i}{i} = riverplot_rect_jj('color', net_cols(i,:), 'position', [x_loc(i) y_loc(i) x_wid(i) y_wid(i)]);
    end

end

net_cols_cell = mat2cell(net_cols, ones(1,size(net_cols,1)), size(net_cols,2));

for cont_i = 1:numel(cont_overlap)

    ribbons = riverplot_ribbon_matrix(ribbon_layers{cont_i}, ribbon_layers{cont_i+1}, cont_overlap{cont_i}, 'colors', net_cols_cell, 'coveragetype', 'relative', 'steepness', 0, 'from_bottom');
    riverplot_toggle_lines(ribbons);
    riverplot_set_ribbon_property(ribbons, 'FaceAlpha', .6);

end

axis off;
set(gca, 'YDir', 'reverse', 'xlim', [0 4*numel(cont_overlap) + 0.5], 'ylim', [0 max(cellfun(@numel, ribbon_layers)) + 0.5]);
set(gcf, 'color', 'w', 'position', [673   422   41+numel(cont_overlap)*327   669]);

end