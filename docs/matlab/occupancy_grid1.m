function occupancy_grid_updated(data_directory, trajectory_path)
% OCCUPANCY_GRID_UPDATED Occupancy grid mapping algorithm based on LiDAR scans (CSV)
%   and estimated trajectories (TXT).
%   This function implements a simple occupancy grid mapping algorithm using
%   LiDAR scan data from a folder of CSV files and robot poses from a TXT file.
%   The map is built using a log-odds representation.
%
%   Input:
%       data_directory (string): Path to the directory containing LiDAR scan CSV files.
%       trajectory_path (string): Path to the text file containing estimated robot trajectory.
%                                 Expected format: Each line has numerical values,
%                                 typically 'timestamp x y z roll pitch yaw' (space-separated).
%                                 We will extract x, y, and yaw (last column for theta).
%                                 If it's just 'x y theta', it will also be handled.
%
%   Example Usage (in MATLAB command window):
%   data_dir = '/content/drive/MyDrive/Harishwar_ICP/SLAM_RVP_Dataset_scans/train00_timestamped_csv';
%   traj_path = '/content/drive/MyDrive/Harishwar_ICP/ICP_Results/estimated_trajectory_icp.txt';
%   occupancy_grid_updated(data_dir, traj_path);
%
%   Author: ChatGPT-like AI (Based on your Python and original MATLAB code)
%   Date: June 17, 2025

% --- Configuration Parameters ---
grid_size = 300;      % Size of the square grid (e.g., 300x300 cells)
map_resolution = 0.05; % Meters per grid cell (e.g., 0.05 m/cell means 15m x 15m map for 300x300 grid)
map_center_x = 0.0;   % World X-coordinate corresponding to the center of the grid
map_center_y = 0.0;   % World Y-coordinate corresponding to the center of the grid

alpha = 0.2;          % Thickness parameter for determining occupied cells (m)

% Log-odds values:
l_0 = 0;              % Initial log-odds (corresponds to P=0.5, unknown)
l_occ = 2;            % Log-odds increment for occupied cells (increases belief)
l_free = -2;          % Log-odds increment for free cells (decreases belief)

% Calculate world extent of the map
map_world_width = grid_size * map_resolution;
map_world_height = grid_size * map_resolution;
map_xmin = map_center_x - map_world_width / 2;
map_ymin = map_center_y - map_world_height / 2;
map_xmax = map_center_x + map_world_width / 2;
map_ymax = map_center_y + map_world_height / 2;

% --- Initialize Grid ---
grid = zeros(grid_size, grid_size);

% --- Get List of Scan Files ---
% Use 'dir' to get a list of CSV files in the data directory
scan_files_struct = dir(fullfile(data_directory, '*.csv'));
if isempty(scan_files_struct)
    error('No CSV files found in the specified data directory: %s', data_directory);
end
scan_filenames = {scan_files_struct.name}; % Extract names from struct array
scan_filenames = sort(scan_filenames); % Sort alphabetically to ensure chronological order

num_total_scans = length(scan_filenames);
% Set a limit for processing (e.g., for testing, or memory management)
% Default to all scans if not specified, or pick a smaller number.
num_scans_to_process_limit = 50; % You can change this
if num_scans_to_process_limit > 0 && num_scans_to_process_limit < num_total_scans
    scan_filenames = scan_filenames(1:num_scans_to_process_limit);
    num_effective_scans = num_scans_to_process_limit;
else
    num_effective_scans = num_total_scans;
end
fprintf('Found %d scan CSV files. Processing %d scans.\n', num_total_scans, num_effective_scans);


% --- ROBOT TRAJECTORY (LOAD FROM PROVIDED TRAJECTORY FILE) ---
robot_poses = [];
try
    % Load trajectory data (assuming space-separated columns)
    % Using dlmread for simplicity with numeric data. For more complex files, use readtable.
    trajectory_data = dlmread(trajectory_path);

    % Determine how to extract x, y, and theta (yaw)
    % Check if it's a column vector (single pose) or matrix
    if iscolumn(trajectory_data) || isrow(trajectory_data) % Single line or single column
        if length(trajectory_data) < 3
            error('Trajectory file has too few columns. Expected at least 3 (x,y,theta) or 7 (timestamp x y z roll pitch yaw).');
        elseif length(trajectory_data) < 7 % Assume x, y, theta
            robot_poses_full = trajectory_data(1:3)'; % Transpose to be row vector if column
        else % Assume timestamp x y z roll pitch yaw
            robot_poses_full = trajectory_data([2, 3, 7])'; % x, y, yaw (1-based indexing for MATLAB)
        end
    else % Matrix
        if size(trajectory_data, 2) < 7
            fprintf('Warning: Trajectory file ''%s'' has %d columns. Assuming format: x y theta directly (e.g., 3 columns).\n', trajectory_path, size(trajectory_data, 2));
            if size(trajectory_data, 2) < 3
                error('Trajectory file has too few columns. Expected at least 3 (x,y,theta) or 7 (timestamp x y z roll pitch yaw).');
            end
            robot_poses_full = trajectory_data(:, 1:3); % Take first 3 columns as x, y, theta
        else
            % Assume 7 columns, extract x, y, and yaw (last column)
            robot_poses_full = trajectory_data(:, [2, 3, 7]); % x, y, yaw (1-based indexing for MATLAB)
        end
    end

    % Ensure the number of poses matches or exceeds the number of scans to process
    if size(robot_poses_full, 1) < num_effective_scans
        fprintf('Warning: Number of poses (%d) in trajectory file is less than number of scans to process (%d). Adjusting num_scans to match poses.\n', size(robot_poses_full, 1), num_effective_scans);
        num_effective_scans = size(robot_poses_full, 1);
        scan_filenames = scan_filenames(1:num_effective_scans); % Truncate scan_filenames list if needed
    elseif size(robot_poses_full, 1) > num_effective_scans
        fprintf('Info: Number of poses (%d) in trajectory file is greater than number of scans to process (%d). Using first %d poses.\n', size(robot_poses_full, 1), num_effective_scans, num_effective_scans);
    end

    robot_poses = robot_poses_full(1:num_effective_scans, :); % Take the necessary number of poses
    fprintf('Loaded %d poses from ''%s''.\n', size(robot_poses, 1), trajectory_path);

catch ME
    fprintf('Error loading trajectory from ''%s'': %s\n', trajectory_path, ME.message);
    fprintf('Falling back to a simple simulated robot trajectory.\n');
    % Fallback to a simple simulation if actual poses are not found or error
    robot_poses = zeros(num_effective_scans, 3);
    for p_idx = 1:num_effective_scans
        r_sim = 5.0;
        t_sim = ((p_idx - 1) / num_effective_scans) * 2 * pi; % MATLAB uses 1-based indexing
        x_sim = r_sim * cos(t_sim);
        y_sim = r_sim * sin(t_sim);
        theta_sim = t_sim + pi / 2;
        robot_poses(p_idx, :) = [x_sim, y_sim, theta_sim];
    end
end

num_scans_to_run = num_effective_scans; % Final number of scans to process


% --- Setup for Real-time Visualization ---
figure(1);
clf; % Clear current figure
ax = gca; % Get current axes
hold(ax, 'on'); % Hold for overlaying robot path

% Use 'imagesc' for grid visualization. 'YDir', 'normal' ensures origin is at bottom-left
% Set X and Y limits in world coordinates for proper scaling.
h_img = imagesc(ax, [map_xmin map_xmax], [map_ymin map_ymax], grid);
set(ax, 'YDir', 'normal'); % Ensure Y-axis is normal (bottom-left origin for grid)
colormap(ax, 'gray'); % Set colormap to grayscale
caxis(ax, [-10 10]); % Set color limits for log-odds display

title(ax, 'Occupancy Grid - Building Map (Log-Odds)');
xlabel(ax, 'World X-coordinate (m)');
ylabel(ax, 'World Y-coordinate (m)');
axis(ax, 'equal'); % Keep aspect ratio for spatial accuracy

% Plot robot path (stores history for visualization)
h_path = plot(ax, [], [], 'r-', 'LineWidth', 2, 'DisplayName', 'Robot Path');
h_robot_dot = plot(ax, [], [], 'ro', 'MarkerSize', 8, 'DisplayName', 'Current Robot Pose');
legend(ax, 'show');

drawnow; % Draw initial plot


grid_half_way = []; % To store grid at half-way point

% --- Process Sensor Data Scan by Scan ---
for p = 1:num_scans_to_run % Loop through each scan
    % Retrieve robot pose for current scan
    current_pose = robot_poses(p, :);
    robot_x = current_pose(1);
    robot_y = current_pose(2);
    robot_theta = current_pose(3);

    % Update robot path visualization
    path_x_data = [get(h_path, 'XData'), robot_x];
    path_y_data = [get(h_path, 'YData'), robot_y];
    set(h_path, 'XData', path_x_data, 'YData', path_y_data);
    set(h_robot_dot, 'XData', robot_x, 'YData', robot_y); % Update current robot dot

    % --- LAZY LOADING OF POINT CLOUD DATA ---
    file_path = fullfile(data_directory, scan_filenames{p});
    try
        % Read CSV. Assumes no headers, just numeric data.
        % If your CSVs have headers, readmatrix might skip the header by default
        % or you might need readtable and then convert to matrix.
        % Based on previous Python analysis, readmatrix should work if it
        % handles column headers correctly.
        current_point_cloud = readmatrix(file_path);

        % --- Process Point Cloud to 2D Ranges (Logic adapted from Python helper) ---
        % Simulate a 2D LiDAR scan from the 3D point cloud
        if isempty(current_point_cloud) || size(current_point_cloud, 1) == 0
            % If no points, return default max ranges for all beams
            ranges = ones(1, 360) * 10.0; % Max range from sensor config
            angle_min_scan = -pi;
            angle_increment_scan = (2*pi)/360;
            range_min_scan = 0.1;
            range_max_scan = 10.0;
        else
            % Filter points within min/max range
            distances_sq = current_point_cloud(:, 1).^2 + current_point_cloud(:, 2).^2;
            distances = sqrt(distances_sq);
            
            valid_points_mask = (distances >= 0.1) & (distances <= 10.0); % Use min/max_range from sensor config
            filtered_points = current_point_cloud(valid_points_mask, :);
            filtered_distances = distances(valid_points_mask);
            
            ranges = ones(1, 360) * 10.0; % Initialize all beams to max_range
            angle_min_scan = -pi; % Assuming 360 deg FOV, centered at 0
            angle_increment_scan = (2*pi)/360; % 1 degree increment

            if ~isempty(filtered_points)
                angles_rad = atan2(filtered_points(:, 2), filtered_points(:, 1)); % atan2(y, x)
                
                % Assign points to angular bins and find the closest point
                for pt_idx = 1:size(filtered_points, 1)
                    angle = angles_rad(pt_idx);
                    dist = filtered_distances(pt_idx);
                    
                    % Map angle to a bin index
                    bin_idx_float = (angle - angle_min_scan) / angle_increment_scan;
                    bin_idx = round(bin_idx_float) + 1; % +1 for MATLAB 1-based indexing
                    
                    % Clamp bin_idx to ensure it's within [1, num_beams]
                    bin_idx = max(1, min(bin_idx, length(ranges)));
                    
                    ranges(bin_idx) = min(ranges(bin_idx), dist);
                end
            end
            range_min_scan = 0.1;
            range_max_scan = 10.0;
        end
        
        scan_data.ranges = ranges;
        scan_data.angle_min = angle_min_scan;
        scan_data.angle_increment = angle_increment_scan;
        scan_data.range_min = range_min_scan;
        scan_data.range_max = range_max_scan;

    catch ME
        fprintf('Could not read or convert %s: %s. Skipping this scan.\n', scan_filenames{p}, ME.message);
        % If scan reading fails, use a dummy "empty" scan (all max_range)
        scan_data.ranges = ones(1, 360) * 10.0;
        scan_data.angle_min = -pi;
        scan_data.angle_increment = (2*pi)/360;
        scan_data.range_min = 0.1;
        scan_data.range_max = 10.0;
    end
    
    beta = scan_data.angle_increment; % Angular width of a single sensor beam (radians)
    angles_for_beams = scan_data.angle_min + (0:length(scan_data.ranges)-1) * scan_data.angle_increment;


    % --- Iterate Through All Grid Cells for Current Scan Update ---
    % Note: Iterating all grid cells for each scan is computationally intensive.
    % For performance, optimize by only checking cells within sensor range.
    for i = 1:grid_size % Rows (grid Y-coordinate)
        for j = 1:grid_size % Columns (grid X-coordinate)

            % Convert grid cell (i, j) to world coordinates (center of cell)
            cell_world_x = map_xmin + (j - 0.5) * map_resolution;
            cell_world_y = map_ymin + (i - 0.5) * map_resolution;

            % Calculate distance from robot to current cell
            r = sqrt((cell_world_x - robot_x)^2 + (cell_world_y - robot_y)^2);

            % Skip if cell is outside general sensor limits for efficiency
            if r > scan_data.range_max || r < scan_data.range_min
                continue;
            end

            % Calculate angle of cell relative to robot's heading
            phi_relative = atan2(cell_world_y - robot_y, cell_world_x - robot_x) - robot_theta;

            % Normalize phi_relative to be within (-pi, pi]
            phi_relative = atan2(sin(phi_relative), cos(phi_relative));

            % Check if cell is within the sensor's FOV (relative to robot's heading)
            % This check is implicit in the k calculation and clamp, but can add for clarity
            % if not (scan_data.angle_min <= phi_relative && phi_relative <= (scan_data.angle_min + (length(scan_data.ranges)-1)*scan_data.angle_increment))
            %     continue;
            % end

            % Find the index 'k' of the sensor beam closest to phi_relative
            k_float = (phi_relative - scan_data.angle_min) / scan_data.angle_increment;
            k = round(k_float) + 1; % +1 for MATLAB 1-based indexing

            % Clamp k to valid indices for 'ranges' array
            k = max(1, min(k, length(scan_data.ranges)));

            % --- Inverse Sensor Model Logic ---
            % If the observed range for beam k is invalid (max range, or too close)
            if scan_data.ranges(k) > scan_data.range_max || scan_data.ranges(k) < scan_data.range_min
                continue; % No information gain
            end

            % If cell is outside the angular width of this beam, no information.
            if abs(phi_relative - angles_for_beams(k)) > beta / 2.0
                continue; % No information gain
            end
            
            % Case 1: Cell is beyond the detected obstacle in this beam (too far)
            if r > scan_data.ranges(k) + alpha / 2.0
                % This cell is definitely not occupied by this scan; it's beyond the obstacle.
                % The standard inverse sensor model sets this to l_0 (unknown) or free if beyond.
                % The original MATLAB code implicitly "passes" (no change) in this specific nested condition.
                % To strictly follow the Python equivalent where "beyond" means no update:
                continue; 
            end

            % Case 2: Cell is close to a detected obstacle (within alpha/2), mark as occupied
            if scan_data.ranges(k) < scan_data.range_max && abs(r - scan_data.ranges(k)) < alpha / 2.0
                grid(i,j) = grid(i,j) + (l_occ - l_0); % Update log-odds for occupied
            % Case 3: Cell is within the free space up to a detected obstacle, mark as free
            elseif r <= scan_data.ranges(k)
                grid(i,j) = grid(i,j) + (l_free - l_0); % Update log-odds for free
            end
        end
    end

    % --- Real-time Display ---
    set(h_img, 'CData', grid); % Update image data
    title(ax, sprintf('Occupancy Grid - Building Map (Log-Odds) - Scan %d/%d', p, num_scans_to_run));
    drawnow limitrate; % Draw plot with rate limit for smoothness

    % Save grid state at roughly half-way point for comparison
    if isempty(grid_half_way) && p >= round(num_scans_to_run / 2.5)
        grid_half_way = grid;
    end
end

% --- Final Processing and Visualization ---
% Convert the log odds to probabilities
final_grid_prob = 1 ./ (1 + exp(-grid)); % Probability P(occupied)

% MATLAB's imagesc naturally plots (row, col) as (y, x).
% The `YDir`, `normal` setting and `extent` should handle orientation.
% If further rotation/transpose is needed for specific visualization, apply here.
% The Python rot90(grid.T, k=2) might be for specific display conventions.
% For MATLAB imagesc with YDir normal and extent, often no additional transpose/rot is needed.

% Initialize if not set (e.g., if num_scans_to_run was very small)
if isempty(grid_half_way)
    grid_half_way = grid; % If less than halfway, just show final grid
end

% Create a new figure for final plots
figure(2);
clf; % Clear previous figure contents

% Plot all images side by side for comparison
subplot(1,3,1)
imagesc([map_xmin map_xmax], [map_ymin map_ymax], grid_half_way);
set(gca, 'YDir', 'normal');
colormap(gca, 'gray');
caxis(gca, [-max(abs(l_occ),abs(l_free))*num_scans_to_run, max(abs(l_occ),abs(l_free))*num_scans_to_run]); % Dynamic color scale
title('Map Half-Way (Log-Odds)');
xlabel('World X-coordinate (m)');
ylabel('World Y-coordinate (m)');
axis equal;

subplot(1,3,2)
imagesc([map_xmin map_xmax], [map_ymin map_ymin], grid);
set(gca, 'YDir', 'normal');
colormap(gca, 'gray');
caxis(gca, [-max(abs(l_occ),abs(l_free))*num_scans_to_run, max(abs(l_occ),abs(l_free))*num_scans_to_run]);
title('Final Map (Log-Odds)');
xlabel('World X-coordinate (m)');
ylabel('World Y-coordinate (m)');
axis equal;


subplot(1,3,3)
imagesc([map_xmin map_xmax], [map_ymin map_ymax], final_grid_prob);
set(gca, 'YDir', 'normal');
colormap(gca, 'gray');
caxis(gca, [0 1]); % Probability always 0 to 1
title('Final Occupancy Map (Probabilities)');
xlabel('World X-coordinate (m)');
ylabel('World Y-coordinate (m)');
axis equal;

% Adjust overall figure layout
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]); % Maximize figure
sgtitle('Occupancy Grid Mapping Results'); % Super title for all subplots

end % End of function