% Functional Brain Reconfiguration During Sustained Pain
% Group consensus community, for Figs. 2-3

%% 1. Basic settings

basedir = '/Volumes/habenula/github/brain_reconfig_pain/'; % set path to local git dir
addpath(genpath(fullfile(basedir, 'functions')));
netdir = fullfile(basedir, 'data/group_consensus_community');

load(fullfile(basedir, 'data/misc/Schaefer_Net_Labels_r263.mat'));
net_cols_orig = Schaefer_Net_Labels.ten_network_col2;
net_cols = net_cols_orig([1 2 7 9 5 6], :);
net_cols_bright = net_cols_orig + 0.05;
net_cols_bright(net_cols_bright > 1) = 1;
Yeo_net_mask = spm_vol(which('Yeo_10networks_4mm.nii'));
Yeo_net_dat = spm_read_vols(Yeo_net_mask);

%% 2. Whole-brain visualization of consensus community

all_task_stage = {'CAPS_alldiv', 'REST_alldiv', ...
    'CAPS_early', 'CAPS_middle', 'CAPS_late'};
idx = 1; % change this index to see different condition or phase

task_stage = split(all_task_stage{idx}, '_');
task_name = task_stage{1};
stage = task_stage{2};

%% 2-1. Draw surface

r = region(fullfile(netdir, sprintf('CAPS2_average_subject_consensus_MNI_task_%s_stage_%s.nii', task_name, stage)), 'unique_mask_values');
out = draw_surface_all(r, net_cols);

view(270, 0); % Left
% view(90, 0); % Right
% view(180, 0); % Anterior

%% 2-2. Draw montage

% Prepare MRIcroGL app, and copy all the clut files from the
% 'brain_reconfig_pain/data/clut' directory (i.e., 'new_net01', ...')
% to the MRIcroGL 'lut' directory.
% (for MacOS, '/Applications/MRIcroGL.app/Contents/Resources/lut').

uimg = which('keuken_2014_enhanced_for_underlay.img');
oimgs = filenames(fullfile(netdir, sprintf('CAPS2_average_subject_consensus_MNI_task_%s_stage_%s_mod_*.nii', task_name, stage)));
colnames = strcat({'new_net'}, num2str([1 2 7 9 5 6].', '%.2d'));
mosaic = 'Z 6'; % Sagittal 1
% mosaic = 'Z -41 41 H 0.4'; % Sagittal 2
% mosaic = 'A -35 -25 -13 5 17 43 60 H 0.16'; % Axial

mricrogl_command = generate_montage_mricrogl(uimg, oimgs, colnames, mosaic);
clipboard('copy', mricrogl_command);

% Start the MRIcroGL app, and click 'Scripting -> New'.
% Paste the copied 'mricrogl_command' to the Scripting window.
% Now click 'Scripting -> Run'.

%% 3. Reconfiguration of brain community structures

all_task_stage = {'CAPS_alldiv', 'REST_alldiv', ...
    'CAPS_early', 'CAPS_middle', 'CAPS_late'};
cont_idx = [1 2]; % change this index to see different condition or phase

cont_dat = NaN(numel(Yeo_net_dat), numel(cont_idx));
for img_i = 1:numel(cont_idx)
    task_stage = split(all_task_stage{cont_idx(img_i)}, '_');
    task_name = task_stage{1};
    stage = task_stage{2};
    img = fullfile(netdir, sprintf('CAPS2_average_subject_consensus_MNI_task_%s_stage_%s.nii', task_name, stage));
    cont_dat(:,img_i) = reshape(spm_read_vols(spm_vol(img)), [], 1);
end

%% 3-1. River plot

wh_cont = all(logical(cont_dat), 2);
cont_overlap = [];
for cont_i = 1:numel(cont_idx)-1
    cont_dat_f = cont_dat(wh_cont,cont_i);
    cont_dat_b = cont_dat(wh_cont,cont_i+1);
    cont_overlap{cont_i} = NaN(max(cont_dat_b), max(cont_dat_f));
    for i = 1:max(cont_dat_f)
        for j = 1:max(cont_dat_b)
            cont_overlap{cont_i}(j,i) = sum(cont_dat_f == i & cont_dat_b == j);
        end
    end
    cont_overlap{cont_i} = cont_overlap{cont_i} ./ sum(cont_overlap{cont_i}, 1:2);
end

draw_riverplot(cont_overlap, net_cols);

%% 3-2. Pie plot

n_net = max(cont_dat(:));
n_ynet = max(Yeo_net_dat(:));
net_overlap_Yeo = NaN(n_ynet, n_net, numel(cont_idx)+1);

for net_i = 1:n_net
    wh_net = cont_dat == net_i;
    wh_net_all = all(wh_net, 2);
    wh_net_each = [wh_net & ~wh_net_all];
    for ynet_i = 1:n_ynet
        wh_ynet = Yeo_net_dat(:) == ynet_i;
        net_overlap_Yeo(ynet_i, net_i, :) = sum([wh_net_all wh_net_each] & wh_ynet);
    end
end

net_i = 1; % change this index to see pie plot of different community
radius_fx = @(x) round(((x / 3000) ^ 0.5) * 600);

for i = 1:numel(cont_idx)+1
    % common, cont1, cont2, ...
    figure;
    wani_pie_(net_overlap_Yeo(:,net_i,i), 'cols', net_cols_bright, 'notext');
    rectangle('Position', [-1.03, -1.03, 2.06, 2.06], 'EdgeColor', [0.8 0.8 0.8 1], 'Linewidth', 8);
    pie_radius = radius_fx(sum(net_overlap_Yeo(:,net_i,i)));
    set(gca, 'units', 'pixel');
    set(gca, 'position', [0 0 1 1] * pie_radius);
    set(gcf, 'position', [0 0 1 1] * pie_radius);
end
