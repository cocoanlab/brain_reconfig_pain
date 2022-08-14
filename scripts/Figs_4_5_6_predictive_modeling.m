% Functional Brain Reconfiguration During Sustained Pain
% Predictive modeling, for Figs. 4-6

%% 1. Basic settings

basedir = '/Volumes/habenula/github/brain_reconfig_pain/'; % set path to local git dir
addpath(genpath(fullfile(basedir, 'functions')));

load(fullfile(basedir, 'data/misc/Schaefer_Net_Labels_r263.mat'));
net_cols = Schaefer_Net_Labels.ten_network_col2;
atlas_regions = Schaefer_Net_Labels.r_2mm;
net_val = Schaefer_Net_Labels.dat(:,2);
net_names = Schaefer_Net_Labels.names;

load(fullfile(basedir, 'data/misc/colormap_jj.mat'));

% Load training data
allegiance_list = filenames(fullfile(basedir, 'data/module_allegiance/module_allegiance_Schaefer263_div*.mat'));
allegiance_caps_div = [];
allegiance_rest_div = [];
for i = 1:numel(allegiance_list)
    load(allegiance_list{i});
    allegiance_caps_div = cat(3, allegiance_caps_div, allegiance_caps);
    allegiance_rest_div = cat(3, allegiance_rest_div, allegiance_rest);
end
allegiance_caps_div = permute(allegiance_caps_div, [1 3 2]);
allegiance_rest_div = permute(allegiance_rest_div, [1 3 2]);
n_div = size(allegiance_caps_div, 2);
n_subj = size(allegiance_caps_div, 3);

load(fullfile(basedir, 'data/predictive_models/caps_rating.mat'));

% Load test data
allegiance_test_list = filenames(fullfile(basedir, 'data/module_allegiance/module_allegiance_test_Schaefer263_div*.mat'));
allegiance_caps_test_div = [];
allegiance_rest_test_div = [];
for i = 1:numel(allegiance_test_list)
    load(allegiance_test_list{i});
    allegiance_caps_test_div = cat(3, allegiance_caps_test_div, allegiance_caps);
    if i<=3; allegiance_rest_test_div = cat(3, allegiance_rest_test_div, allegiance_rest); end
end
allegiance_caps_test_div = permute(allegiance_caps_test_div, [1 3 2]);
allegiance_rest_test_div = permute(allegiance_rest_test_div, [1 3 2]);
n_test_div = size(allegiance_caps_test_div, 2);
n_test_subj = size(allegiance_caps_test_div, 3);

load(fullfile(basedir, 'data/predictive_models/caps_rating_test.mat'));

gray_mask = spm_vol(which('Schaefer_265_combined_2mm.nii'));
gray_dat = spm_read_vols(gray_mask);
wh_gray = logical(gray_dat(:));
n_brain = sum(wh_gray);

%% 2. Group-level module allegiance

allegiance_caps_grpmean = mean(allegiance_caps_div, 2:3);
allegiance_rest_grpmean = mean(allegiance_rest_div, 2:3);

vis_corr(reformat_r_new(allegiance_caps_grpmean, 'reconstruct'), ...
    'nolines', 'group', net_val, 'group_color', net_cols, 'group_linewidth', 2, 'group_linecolor', 'k', 'smooth', 'clim', [0 0.45], 'colors', col_diff_map1, ...
    'group_tick', 'group_tickstyle', 'edge', 'group_tickwidth', 1.5, 'group_ticklength', 5, 'group_tickoffset', 1, 'triangle', 'triangle_width', 4, 'triangle_color', col_posneg(1,:), 'colorbar');
vis_corr(reformat_r_new(allegiance_rest_grpmean, 'reconstruct'), ...
    'nolines', 'group', net_val, 'group_color', net_cols, 'group_linewidth', 2, 'group_linecolor', 'k', 'smooth', 'clim', [0 0.45], 'colors', col_diff_map1, ...
    'group_tick', 'group_tickstyle', 'edge', 'group_tickwidth', 1.5, 'group_ticklength', 5, 'group_tickoffset', 1, 'triangle', 'triangle_width', 4, 'triangle_color', col_posneg(2,:), 'colorbar');

%% 3. SVM

%%%% Training %%%%
% dat = fmri_data;
% dat.dat = [squeeze(mean(allegiance_caps_div, 2)), squeeze(mean(allegiance_rest_div, 2))];
% dat.Y = [ones(n_subj, 1); -ones(n_subj, 1)];
% wh_fold = [1:n_subj 1:n_subj];
% 
% [pred_model.cverr, pred_model.stats, pred_model.optout] = predict(dat, 'algorithm_name', 'cv_svm', 'nfolds', wh_fold, 'error_type', 'mcr');

%%%% Bootstrap %%%%
% bootdat = fmri_data;
% bootnum = 10000;
% bootW = zeros(size(pred_model.optout{1}, 1), bootnum);
% bootsamp = randi(n_subj, n_subj, bootnum);
% for boot_i = 1:bootnum
%     fprintf('Working on bootstrap %.6d ...\n', boot_i);
%     bootdat.dat = [squeeze(mean(allegiance_caps_div(:,:,bootsamp(:,boot_i)), 2)), squeeze(mean(allegiance_rest_div(:,:,bootsamp(:,boot_i)), 2))];
%     bootdat.Y = [ones(n_subj, 1); -ones(n_subj, 1)];
%     [~, ~, tempw] = predict(bootdat, 'verbose', 0, 'algorithm_name', 'cv_svm', 'nfolds', 1, 'error_type', 'mcr', 'noparallel', 'onlyboot');
%     bootW(:, boot_i) = tempw{1};
% end
% bootW_mean = mean(bootW, 2);
% bootW_std = std(bootW, [], 2);
% bootW_std(bootW_std == 0) = Inf;
% bootW_z = bootW_mean ./ bootW_std;
% bootW_p = 2 * (1 - normcdf(abs(bootW_z)));

%%%% Load trained SVM model %%%%
load(fullfile(basedir, 'data/predictive_models/SVM_model_Schaefer.mat'));
load(fullfile(basedir, 'data/predictive_models/SVM_model_Schaefer_boot.mat'));

%% 3-1. Visualize model weight

pred_model_w = reformat_r_new(pred_model.optout{1}, 'reconstruct');
vis_corr(pred_model_w, 'nolines', 'group', net_val, 'group_color', net_cols, 'group_linewidth', 2, 'group_linecolor', 'k', 'smooth', 'colors', col_diff_map1, ...
    'group_tick', 'group_tickstyle', 'edge', 'group_tickwidth', 1.5, 'group_ticklength', 5, 'group_tickoffset', 1, 'triangle', 'no_triangle_line', 'colorbar');

%% 3-2. Classification accuracy: Training

dist_hp = pred_model.stats.dist_from_hyperplane_xval;
dist_hp = reshape(dist_hp, [], 2);

roc_result = roc_plot(dist_hp(:), [ones(size(dist_hp,1),1); zeros(size(dist_hp,1),1)], 'color', 'r', 'twochoice');
set(gca, 'fontsize', 18, 'linewidth', 2, 'ticklength', [.02 .02], 'tickdir', 'out');
set(gcf, 'color', 'w', 'position', [1000         731         285         254]);
box off;

out = plot_specificity_box(dist_hp(:,1), dist_hp(:,2), 'color', col_crtwrg);
xticklabels({'Caps', 'Cont'});
set(gcf, 'position', [655   620   210   208]);

%% 3-3. Classification accuracy: Test

dist_hp = [sum(squeeze(mean(allegiance_caps_test_div,2)) .* pred_model.optout{1}, 'omitnan')' + pred_model.optout{3}, ...
    sum(squeeze(mean(allegiance_rest_test_div,2)) .* pred_model.optout{1}, 'omitnan')' + pred_model.optout{3}];

roc_result = roc_plot(dist_hp(:), [ones(size(dist_hp,1),1); zeros(size(dist_hp,1),1)], 'color', 'r', 'twochoice');
set(gca, 'fontsize', 18, 'linewidth', 2, 'ticklength', [.02 .02], 'tickdir', 'out');
set(gcf, 'color', 'w', 'position', [1000         731         285         254]);
box off;

out = plot_specificity_box(dist_hp(:,1), dist_hp(:,2), 'color', col_crtwrg);
xticklabels({'Caps', 'Cont'});
set(gcf, 'position', [655   620   210   208]);

%% 3-4. Thresholded weight circos plot

pred_model_thr_w = reformat_r_new(pred_model.optout{1} .* double(bootW_p <= FDR(bootW_p, 0.05)), 'reconstruct');

[~, ~, surv_w] = find(triu(pred_model_thr_w, 1));
fx = @(x) ((abs(x) - min(abs(x))) ./ (max(abs(x)) - min(abs(x)))) .^ 2.5;
thr_alpha = fx(surv_w);
fx = @(x) (abs(x) - min(abs(x))) ./ (max(abs(x)) - min(abs(x))) * 2.25 + 0.25;
thr_width = fx(surv_w);
thr_color = zeros(numel(surv_w), 3);
thr_color(surv_w > 0, :) = repmat([255,0,0] ./ 255, sum(surv_w > 0), 1);
thr_color(surv_w < 0, :) = repmat([10,150,255] ./ 255, sum(surv_w < 0), 1);

circos_multilayer(pred_model_thr_w, 'group', net_val, 'group_color', net_cols, 'length_ratio', [10 0 10], 'patch_edge_alpha', 0, ...
    'conn_color', thr_color, 'conn_alpha', thr_alpha, 'conn_width', thr_width);

set(gcf, 'position', [560    50   953   898]);

%% 3-5. Top 50 stable weights, glass brain

pred_model_thr_w_top50 = reformat_r_new(pred_model.optout{1} .* double(bootW_p < prctile(bootW_p, 100 * 50/numel(bootW_p))), 'reconstruct');

glass_w = pred_model_thr_w_top50;
glass_w = (abs((glass_w ./ max(abs(glass_w(:)))) .^ 3) * 0.8 + 0.2) .* sign(glass_w);
out = glass_brain_network(atlas_regions, 'group', net_val, 'colors', net_cols, ...
    'edge_weights', glass_w, 'edge_alpha', 0.7, 'pos_edge_color', col_posneg(1,:), 'neg_edge_color', col_posneg(2,:), ...
    'hl_node_edge', 2.5, 0.5, [.8 .8 .8], 'norm_factor', 1/3, 'cortex_alpha', .05, 'cerebellum_alpha', .1);
set(gcf, 'Position', [1           6        1027         949]);

view(90, 0); % Right
% view(0, 90); % Superior
% view(0, 0); % Posterior

%% 3-6. Seed-based allegiance from hub regions

pred_model_thr_w_unc = reformat_r_new(pred_model.optout{1} .* double(bootW_p < 0.05), 'reconstruct');
[~, pos_hub_reg] = max(sum(pred_model_thr_w_unc .* double(pred_model_thr_w_unc > 0)));
[~, neg_hub_reg] = min(sum(pred_model_thr_w_unc .* double(pred_model_thr_w_unc < 0)));
fprintf('Largest wegithed degree centrality\nPositive: Region #%d, %s\nNegative: Region #%d, %s\n', pos_hub_reg, net_names{pos_hub_reg}, neg_hub_reg, net_names{neg_hub_reg});

posnegall = load(fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d.mat', pos_hub_reg)));

%% 3-6-1. Mean seed-based allegiance map

posneg_hub_caps_img = gray_mask;
posneg_hub_caps_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_alldiv.nii', pos_hub_reg));
posneg_hub_caps_dat = gray_dat;
posneg_hub_caps_dat(wh_gray) = mean(posnegall.allegiance_caps_reg_alldiv, 2);
spm_write_vol(posneg_hub_caps_img, posneg_hub_caps_dat);

posneg_hub_rest_img = gray_mask;
posneg_hub_rest_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_REST_stage_alldiv.nii', pos_hub_reg));
posneg_hub_rest_dat = gray_dat;
posneg_hub_rest_dat(wh_gray) = mean(posnegall.allegiance_rest_reg_alldiv, 2);
spm_write_vol(posneg_hub_rest_img, posneg_hub_rest_dat);

%% 3-6-2. Thresholded map

[~,p,~,stat] = ttest(posnegall.allegiance_caps_reg_alldiv.', posnegall.allegiance_rest_reg_alldiv.');
posneg_hub_diff_img = gray_mask;
posneg_hub_diff_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_contrast_CAPS_alldiv_vs_REST_alldiv.nii', pos_hub_reg));
posneg_hub_diff_dat = gray_dat;
posneg_hub_diff_dat(wh_gray) = sum(p <= [0.05, 0.01, FDR(p, 0.05)].') .* sign(stat.tstat);
spm_write_vol(posneg_hub_diff_img, posneg_hub_diff_dat);

posneg_hub_diff_reg = region(posneg_hub_diff_img.fname);
posneg_hub_diff_reg = posneg_hub_diff_reg(cellfun(@(x) max(abs(x)), {posneg_hub_diff_reg.val}) == 3);
posneg_hub_diff_reg = region2imagevec(posneg_hub_diff_reg);
posneg_hub_diff_reg.fullpath = posneg_hub_diff_img.fname;
write(posneg_hub_diff_reg, 'overwrite');

%% 3-6-3. Draw montage

% Prepare MRIcroGL app, and copy all the clut files from the
% 'brain_reconfig_pain/data/clut' directory (i.e., 'new_net01', ...')
% to the MRIcroGL 'lut' directory.
% (for MacOS, '/Applications/MRIcroGL.app/Contents/Resources/lut').

uimg = which('keuken_2014_enhanced_for_underlay.img');
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_alldiv.nii', pos_hub_reg))}; % mean for CAPS
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_REST_stage_alldiv.nii', pos_hub_reg))}; % mean for REST
oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_contrast_CAPS_alldiv_vs_REST_alldiv.nii', pos_hub_reg))}; % thresholded
colnames = {'jj'};
mosaic = 'Z 4 27 39 H 0.48'; % Sagittal
% mosaic = 'C -26 44 H 0.2'; % Coronal
% mosaic = 'A 2 63 H 0.25'; % Axial
% clim = [0 0.45]; % for mean allegiance image
clim = [-3 3]; % for thresholded image
    
mricrogl_command = generate_montage_mricrogl(uimg, oimgs, colnames, mosaic, 'clim', clim);
clipboard('copy', mricrogl_command);

% Start the MRIcroGL app, and click 'Scripting -> New'.
% Paste the copied 'mricrogl_command' to the Scripting window.
% Now click 'Scripting -> Run'.

%% 4. PCR

%%%% Training %%%%
% dat = fmri_data;
% dat.dat = reshape(allegiance_caps_div, size(allegiance_caps_div,1), []);
% dat.Y = caps_rating(:);
% wh_fold = repmat(1:n_subj, n_div, 1);
% wh_fold = wh_fold(:);
% 
% pred_out_corr = [];
% numc = 1:469;
% for numc_i = 1:numel(numc)
%     [~, stats, ~] = predict(dat, 'algorithm_name', 'cv_pcr', 'numcomponents', numc(numc_i), 'nfolds', wh_fold, 'error_type', 'mse');
%     pred_out_corr(:, numc_i) = multicorr(reshape(stats.Y, n_div, n_subj), reshape(stats.yfit, n_div, n_subj));
% end
% [~, best_numc] = max(mean(pred_out_corr));
% 
% [pred_model.cverr, pred_model.stats, pred_model.optout] = predict(dat, 'algorithm_name', 'cv_pcr', 'numcomponents', best_numc, 'nfolds', wh_fold, 'error_type', 'mse');

%%%% Bootstrap %%%%
% bootdat = fmri_data;
% bootnum = 10000;
% bootW = zeros(size(pred_model.optout{1}, 1), bootnum);
% bootsamp = randi(n_subj, n_subj, bootnum);
% for boot_i = 1:bootnum
%     fprintf('Working on bootstrap %.6d ...\n', boot_i);
%     bootdat.dat = reshape(allegiance_caps_div(:,:,bootsamp(:,boot_i)), size(allegiance_caps_div,1), []);
%     bootdat.Y = reshape(caps_rating(:,bootsamp(:,boot_i)), [], 1);
%     [~, ~, tempw] = predict(bootdat, 'verbose', 0, 'algorithm_name', 'cv_pcr', 'numcomponents', best_numc, 'nfolds', 1, 'error_type', 'mse', 'noparallel', 'onlyboot');
%     bootW(:, boot_i) = tempw{1};
% end
% bootW_mean = mean(bootW, 2);
% bootW_std = std(bootW, [], 2);
% bootW_std(bootW_std == 0) = Inf;
% bootW_z = bootW_mean ./ bootW_std;
% bootW_p = 2 * (1 - normcdf(abs(bootW_z)));

%%%% Load trained PCR model %%%%
load(fullfile(basedir, 'data/predictive_models/PCR_model_Schaefer.mat'));
load(fullfile(basedir, 'data/predictive_models/PCR_model_Schaefer_boot.mat'));

%% 4-1. Visualize model weight

pred_model_w = reformat_r_new(pred_model.optout{1}, 'reconstruct');
vis_corr(pred_model_w, 'nolines', 'group', net_val, 'group_color', net_cols, 'group_linewidth', 2, 'group_linecolor', 'k', 'smooth', 'colors', col_diff_map1, ...
    'group_tick', 'group_tickstyle', 'edge', 'group_tickwidth', 1.5, 'group_ticklength', 5, 'group_tickoffset', 1, 'triangle', 'no_triangle_line', 'colorbar');

%% 4-2. Prediction vs. Outcome: Training

yval = reshape(pred_model.stats.Y, n_div, n_subj);
yfit = reshape(pred_model.stats.yfit, n_div, n_subj);

plot_y_yfit_jj(yval, yfit, 'color', col_diff_map2, 'clim', [-1 1], 'data_alpha', 0.9, 'line_alpha', 0.9, 'ylim', [-0.1 1]);

[~, bootr] = bootstrap_corr(multicorr(yval, yfit), 10000, 'only_p', 'rng', 2020);

%% 4-3. Prediction vs. Outcome: Test

yval = caps_rating_test;
yfit = sum(reshape(allegiance_caps_test_div, size(allegiance_caps_test_div,1), []) .* pred_model.optout{1}, 'omitnan') + pred_model.optout{2};
yfit = reshape(yfit, n_test_div, n_test_subj);

plot_y_yfit_jj(yval, yfit, 'color', col_diff_map2, 'clim', [-1 1], 'data_alpha', 0.9, 'line_alpha', 0.9, 'ylim', [-0.1 1]);

[~, bootr] = bootstrap_corr(multicorr(yval, yfit), 10000, 'only_p', 'rng', 2020);

%% 4-4. Thresholded weight circos plot

pred_model_thr_w = reformat_r_new(pred_model.optout{1} .* double(bootW_p < 0.05), 'reconstruct');

[~, ~, surv_w] = find(triu(pred_model_thr_w, 1));
surv_w_p = bootW_p(bootW_p < 0.05);
wh_pos_fdr = surv_w > 0 & surv_w_p <= FDR(bootW_p, 0.05);
wh_neg_fdr = surv_w < 0 & surv_w_p <= FDR(bootW_p, 0.05);
wh_pos_unc = surv_w > 0 & surv_w_p > FDR(bootW_p, 0.05);
wh_neg_unc = surv_w < 0 & surv_w_p > FDR(bootW_p, 0.05);

thr_cols = zeros(numel(surv_w), 3);
thr_cols(wh_pos_fdr, :) = repmat(col_posneg2(1,:), sum(wh_pos_fdr), 1);
thr_cols(wh_neg_fdr, :) = repmat(col_posneg2(2,:), sum(wh_neg_fdr), 1);
thr_cols(wh_pos_unc, :) = repmat(col_posneg2(3,:), sum(wh_pos_unc), 1);
thr_cols(wh_neg_unc, :) = repmat(col_posneg2(4,:), sum(wh_neg_unc), 1);

thr_width = zeros(numel(surv_w), 1);
fx = @(x) (abs(x) - min(abs(x))) ./ (max(abs(x)) - min(abs(x))) * 2 + 1;
thr_width(wh_pos_fdr | wh_neg_fdr) = fx(surv_w(wh_pos_fdr | wh_neg_fdr));
fx = @(x) (abs(x) - min(abs(x))) ./ (max(abs(x)) - min(abs(x))) * 0.5 + 0.5;
thr_width(wh_pos_unc | wh_neg_unc) = fx(surv_w(wh_pos_unc | wh_neg_unc));

thr_alpha = zeros(numel(surv_w), 1);
fx = @(x) 0.6 + 0.4 * ((abs(x) - min(abs(x))) ./ (max(abs(x)) - min(abs(x)))) .^ 2;
thr_alpha(wh_pos_fdr | wh_neg_fdr) = fx(surv_w(wh_pos_fdr | wh_neg_fdr));
fx = @(x) 0.6 * ((abs(x) - min(abs(x))) ./ (max(abs(x)) - min(abs(x)))) .^ 2;
thr_alpha(wh_pos_unc | wh_neg_unc) = fx(surv_w(wh_pos_unc | wh_neg_unc));

thr_order = [find(wh_pos_unc | wh_neg_unc); find(wh_pos_fdr | wh_neg_fdr)];

circos_multilayer(pred_model_thr_w, 'group', net_val, 'group_color', net_cols, 'length_ratio', [10 0 10], 'patch_edge_alpha', 0, ...
    'conn_color', thr_cols, 'conn_width', thr_width, 'conn_alpha', thr_alpha, 'conn_order', thr_order);

set(gcf, 'position', [560    50   953   898]);

%% 4-5. Top 50 stable weights, glass brain

pred_model_thr_w_top50 = reformat_r_new(pred_model.optout{1} .* double(bootW_p < prctile(bootW_p, 100 * 50/numel(bootW_p))), 'reconstruct');

glass_w = pred_model_thr_w_top50;
glass_w = (abs((glass_w ./ max(abs(glass_w(:)))) .^ 3) * 0.8 + 0.2) .* sign(glass_w);
out = glass_brain_network(atlas_regions, 'group', net_val, 'colors', net_cols, ...
    'edge_weights', glass_w, 'edge_alpha', 0.7, 'pos_edge_color', col_posneg(1,:), 'neg_edge_color', col_posneg(2,:), ...
    'hl_node_edge', 2.5, 0.5, [.8 .8 .8], 'norm_factor', 1/3, 'cortex_alpha', .05, 'cerebellum_alpha', .1);
set(gcf, 'Position', [1           6        1027         949]);

view(90, 0); % Right
% view(0, 90); % Superior
% view(0, 0); % Posterior

%% 4-6. Seed-based allegiance map

pred_model_thr_w_unc = reformat_r_new(pred_model.optout{1} .* double(bootW_p < 0.05), 'reconstruct');
[~, pos_hub_reg] = max(sum(pred_model_thr_w_unc .* double(pred_model_thr_w_unc > 0)));
[~, neg_hub_reg] = min(sum(pred_model_thr_w_unc .* double(pred_model_thr_w_unc < 0)));
fprintf('Largest wegithed degree centrality\nPositive: Region #%d, %s\nNegative: Region #%d, %s\n', pos_hub_reg, net_names{pos_hub_reg}, neg_hub_reg, net_names{neg_hub_reg});

posall = load(fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d.mat', pos_hub_reg)));
negall = load(fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d.mat', neg_hub_reg)));

%% 4-6-1. Mean seed-based allegiance map

pos_hub_early_img = gray_mask;
pos_hub_early_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_early.nii', pos_hub_reg));
pos_hub_early_dat = gray_dat;
pos_hub_early_dat(wh_gray) = mean(posall.allegiance_caps_reg_early, 2);
spm_write_vol(pos_hub_early_img, pos_hub_early_dat);

pos_hub_middle_img = gray_mask;
pos_hub_middle_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_middle.nii', pos_hub_reg));
pos_hub_middle_dat = gray_dat;
pos_hub_middle_dat(wh_gray) = mean(posall.allegiance_caps_reg_middle, 2);
spm_write_vol(pos_hub_middle_img, pos_hub_middle_dat);

pos_hub_late_img = gray_mask;
pos_hub_late_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_late.nii', pos_hub_reg));
pos_hub_late_dat = gray_dat;
pos_hub_late_dat(wh_gray) = mean(posall.allegiance_caps_reg_late, 2);
spm_write_vol(pos_hub_late_img, pos_hub_late_dat);

neg_hub_early_img = gray_mask;
neg_hub_early_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_early.nii', neg_hub_reg));
neg_hub_early_dat = gray_dat;
neg_hub_early_dat(wh_gray) = mean(negall.allegiance_caps_reg_early, 2);
spm_write_vol(neg_hub_early_img, neg_hub_early_dat);

neg_hub_middle_img = gray_mask;
neg_hub_middle_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_middle.nii', neg_hub_reg));
neg_hub_middle_dat = gray_dat;
neg_hub_middle_dat(wh_gray) = mean(negall.allegiance_caps_reg_middle, 2);
spm_write_vol(neg_hub_middle_img, neg_hub_middle_dat);

neg_hub_late_img = gray_mask;
neg_hub_late_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_late.nii', neg_hub_reg));
neg_hub_late_dat = gray_dat;
neg_hub_late_dat(wh_gray) = mean(negall.allegiance_caps_reg_late, 2);
spm_write_vol(neg_hub_late_img, neg_hub_late_dat);

%% 4-6-2. Thresholded map

[~,p,~,stat] = ttest(posall.allegiance_caps_reg_late.', posall.allegiance_caps_reg_early.');
pos_hub_diff_img = gray_mask;
pos_hub_diff_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_contrast_CAPS_late_vs_CAPS_early.nii', pos_hub_reg));
pos_hub_diff_dat = gray_dat;
pos_hub_diff_dat(wh_gray) = sum(p <= [0.05, 0.01, FDR(p, 0.05)].') .* sign(stat.tstat);
spm_write_vol(pos_hub_diff_img, pos_hub_diff_dat);

pos_hub_diff_reg = region(pos_hub_diff_img.fname);
pos_hub_diff_reg = pos_hub_diff_reg(cellfun(@(x) max(abs(x)), {pos_hub_diff_reg.val}) == 3);
pos_hub_diff_reg = region2imagevec(pos_hub_diff_reg);
pos_hub_diff_reg.fullpath = pos_hub_diff_img.fname;
write(pos_hub_diff_reg, 'overwrite');

[~,p,~,stat] = ttest(negall.allegiance_caps_reg_late.', negall.allegiance_caps_reg_early.');
neg_hub_diff_img = gray_mask;
neg_hub_diff_img.fname = fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_contrast_CAPS_late_vs_CAPS_early.nii', neg_hub_reg));
neg_hub_diff_dat = gray_dat;
neg_hub_diff_dat(wh_gray) = sum(p <= [0.05, 0.01, FDR(p, 0.05)].') .* sign(stat.tstat);
spm_write_vol(neg_hub_diff_img, neg_hub_diff_dat);

neg_hub_diff_reg = region(neg_hub_diff_img.fname);
neg_hub_diff_reg = neg_hub_diff_reg(cellfun(@(x) max(abs(x)), {neg_hub_diff_reg.val}) == 3);
neg_hub_diff_reg = region2imagevec(neg_hub_diff_reg);
neg_hub_diff_reg.fullpath = neg_hub_diff_img.fname;
write(neg_hub_diff_reg, 'overwrite');

%% 4-6-3. Draw montage

% Prepare MRIcroGL app, and copy all the clut files from the
% 'brain_reconfig_pain/data/clut' directory (i.e., 'new_net01', ...')
% to the MRIcroGL 'lut' directory.
% (for MacOS, '/Applications/MRIcroGL.app/Contents/Resources/lut').

uimg = which('keuken_2014_enhanced_for_underlay.img');
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_early.nii', pos_hub_reg));}; % mean for early, pos
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_middle.nii', pos_hub_reg));}; % mean for middle, pos
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_middle.nii', pos_hub_reg));}; % mean for late, pos
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_early.nii', neg_hub_reg));}; % mean for early, neg
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_middle.nii', neg_hub_reg));}; % mean for middle, neg
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_groupmean_task_CAPS_stage_middle.nii', neg_hub_reg));}; % mean for late, neg
% oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_contrast_CAPS_late_vs_CAPS_early.nii', pos_hub_reg))}; % thresholded, pos
oimgs = {fullfile(basedir, 'data/module_allegiance', sprintf('module_allegiance_Schaefer263_reg_%.3d_contrast_CAPS_late_vs_CAPS_early.nii', neg_hub_reg))}; % thresholded, neg
colnames = {'jj'};
% mosaic = 'Z 6'; % Sagittal, pos
% mosaic = 'C -26 44 H 0.2'; % Coronal, pos
% mosaic = 'A -43 -25 9 34 53 H 0.16'; % Axial, pos
% mosaic = 'Z 4 27 39 H 0.48'; % Sagittal, neg
mosaic = 'C -19 10 40 H 0.18'; % Coronal, neg
% mosaic = 'A 9'; % Axial, neg
% clim = [0.1 0.4]; % for mean allegiance image
clim = [-3 3]; % for thresholded image

mricrogl_command = generate_montage_mricrogl(uimg, oimgs, colnames, mosaic, 'clim', clim);
clipboard('copy', mricrogl_command);

% Start the MRIcroGL app, and click 'Scripting -> New'.
% Paste the copied 'mricrogl_command' to the Scripting window.
% Now click 'Scripting -> Run'.
