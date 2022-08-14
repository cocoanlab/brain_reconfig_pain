function out = plot_y_yfit_jj(yval, yfit, varargin)

xlim = [-.1 1.2];
ylim = [];
data_alpha  = 1;
line_alpha  = 1;
dotsize = 40;
do_xyline = false;
colors = [255,237,160
    254,217,118
    254,178,76
    253,141,60
    252,78,42
    227,26,28
    189,0,38
    189,0,38
    189,0,38
    189,0,38
    189,0,38
    189,0,38] ./ 255;
clim = [-1 1];

for i = 1:length(varargin)
    if ischar(varargin{i})
        switch varargin{i}
            % functional commands
            case {'xlim'}
                xlim = varargin{i+1};
            case {'ylim'}
                ylim = varargin{i+1};
            case {'data_alpha'}
                data_alpha = varargin{i+1};
            case {'line_alpha'}
                line_alpha = varargin{i+1};
            case {'dotsize'}
                dotsize = varargin{i+1};
            case {'xyline'}
                do_xyline = true;
            case {'color'}
                colors = varargin{i+1};
            case {'clim'}
                clim = varargin{i+1};
        end
    end
end

%% plotting

create_figure('predicted');

clear test_por;
if iscell(yval)
    yval = cat(2, yval{:});
end
if iscell(yfit)
    yfit = cat(2, yfit{:});
end

[obs_num, subj_num] = size(yval);

marker_shapes = repmat('osd^v><', 1, 40);
col_parts = linspace(clim(1)-eps, clim(2)+eps, size(colors,1)+1);
col_parts([1 end]) = [-Inf Inf];

for i = 1:subj_num
    x = yval(:,i);
    y = yfit(:,i);
    b = glmfit(x,y);
    out.b(i,1) = b(2);
    out.r(i,1) = corr(x,y);
    line_colors(i,:) = colors(sum(out.r(i) >= col_parts), :);
end

if mean(out.r) > 0
    [~, draw_idx] = sort(out.r, 'ascend');
else
    [~, draw_idx] = sort(out.r, 'descend');
end

for i = draw_idx'
    x = yval(:,i);
    y = yfit(:,i);
    b = glmfit(x,y);
    hold on;
    line_h(i) = line(xlim, b'*[ones(1,2); xlim], 'linewidth', 1.5, 'color', line_colors(i,:));
    line_h(i).Color(4) = line_alpha;
    h = scatter(x, y, dotsize, line_colors(i,:), 'filled', 'markerfacealpha', data_alpha, 'marker', marker_shapes(i));
end

if do_xyline
    line(xlim, xlim, 'linewidth', 4, 'linestyle', ':', 'color', [.5 .5 .5]);
end

set(gcf, 'position', [1   747   218   208]);
if isempty(ylim)
    ylim = [min(yfit(:)), max(yfit(:))];
    ylim = [ylim(1) - diff(ylim) * 0.15, ylim(2) + diff(ylim) * 0.15];
end
set(gca, 'tickdir', 'out', 'TickLength', [.03 .03], 'linewidth', 1.5, 'xlim', xlim, 'ylim', ylim, 'fontsize', 18);
end