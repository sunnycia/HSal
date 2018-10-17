function other_map = compute_othermap(m,fixation_dir,row,column)
fixation_map_list = dir(fixation_dir);
total_fixation = length(fixation_map_list)-2;

% generate random m different number

rdm_idx = randperm(total_fixation,m);

other_map = zeros(row, column);
for j = 1 : m
    idx = rdm_idx(j) + 2;
    % fix_name = strcat('fixation', num2str(idx),'.mat');
    % load([fixationPts filesep fix_name])
    fixation_map_path = fullfile(fixation_dir, fixation_map_list(idx).name);
    fixation=imread(fixation_map_path);
    fixation(fixation<255)=0;
    fixation=im2double(fixation);
    other_map = other_map + fixation;
end