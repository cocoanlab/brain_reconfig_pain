function out = draw_surface_all(r, cols)

r_cols = cols(1:numel(r), :);

depth = 3;

numVox = cat(1,r(:).numVox);
poscm = [repelem(r_cols(:,1), numVox) repelem(r_cols(:,2), numVox) repelem(r_cols(:,3), numVox)];

figure;
[out.h_surf_L, out.colorbar] = cluster_surf_cocoan(r, 'underlay', 'fsavg_left', 'depth', depth, 'colormaps', poscm, [], 'prioritize_last', true);
[out.h_surf_R, out.colorbar] = cluster_surf_cocoan(r, 'underlay', 'fsavg_right', 'depth', depth, 'colormaps', poscm, [], 'prioritize_last', true);
out.h = get(gca, 'children');
set(out.h(2), 'BackFaceLighting', 'lit');
set(out.h(3), 'BackFaceLighting', 'reverselit');
axis vis3d;
view(180, 0);

end