clear; clc; close all;
% Config
Fs = 100;
T = 60;
% Split
splits = struct();
splits(1).name = 'train';
splits(1).count = 100;
splits(2).name = 'val';
splits(2).count = 20;
splits(3).name = 'test';
splits(3).count = 20;
% Check if 'data_no_bias' directory exists
output_dir = 'data_no_bias';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created directory: %s\n', output_dir);
end
% --- Main Execution Loop ---
for i = 1:length(splits)
    split_name = splits(i).name;
    num_sims = splits(i).count;
    
    fprintf('Generating %s set (%d simulations)...\n', split_name, num_sims);
    
    % Call function
    [inputs, targets] = generate_dataset(num_sims, Fs, T);
    
    % Cfilenames
    input_filename = fullfile(output_dir, sprintf('%s_inputs.csv', split_name));
    target_filename = fullfile(output_dir, sprintf('%s_targets.csv', split_name));
    
    % Save CSV
    writematrix(inputs, input_filename);
    writematrix(targets, target_filename);
    
    fprintf('Saved: %s and %s\n', input_filename, target_filename);
end
fprintf('All data generation complete.\n');

%Genreration function
function [all_inputs, all_targets] = generate_dataset(num_simulations, Fs, T)
    dt = 1/Fs;
    t = 0:dt:T;
    
    all_inputs = [];
    all_targets = [];
    for i = 1:num_simulations
        % 1. Create Random Position Trajectory
        wx = 0.5 + 0.2*rand; 
        wy = 0.5 + 0.2*rand;
        
        x = 5 * sin(wx*t);
        y = 5 * sin(wy*t);
        z = 2 * sin(0.5*t);
        pos = [x', y', z'];
        
        % 2. Create Random Orientation Trajectory
        roll_amp = 10 + 10*rand; % Random amplitude (10 to 20 deg)
        pitch_amp = 10 + 10*rand;% Random amplitude (10 to 20 deg)
        
        roll_motion = roll_amp * sind(0.8*t); 
        pitch_motion = pitch_amp * sind(0.8*t);
        yaw_motion = 45 * sind(0.5*t) + 0.5*t * (1 + 0.1*rand); % Add some yaw drift
        
        % Assemble Euler Angles: [Yaw, Pitch, Roll]
        eul = [yaw_motion', pitch_motion', roll_motion']; 
        
        % Convert Euler (ZYX convention) to quaternion array
        q_eul = eul2quat(eul, 'ZYX');
        
        % Convert Quaternions to Rotation Matrices (3x3xM)
        rotm_eul = quat2rotm(q_eul);
        
        % Create Trajectory with both Position and Orientation
        trajectory = waypointTrajectory(pos, t, 'SampleRate', Fs, 'Orientation', rotm_eul);
        [p_true, q_true, v_true, a_true, w_true] = lookupPose(trajectory, t);
        
        % 2. Simulate IMU WITHOUT BIAS (No bias since we want to compair the filter itself)
        imu = imuSensor('accel-gyro-mag', 'SampleRate', Fs);
        
        % set MagneticField
        imu.MagneticField = [20 0 -40]; 
        
        % set bias (zero)
        imu.Gyroscope.ConstantBias = [0 0 0];
        imu.Accelerometer.ConstantBias = [0 0 0];
        
        % set noise
        imu.Accelerometer.NoiseDensity = 0.002;
        imu.Gyroscope.NoiseDensity = 0.0001; 
        imu.Magnetometer.NoiseDensity = 0.1;
        
        % Generate Data
        [accel_meas, gyro_meas, mag_meas] = imu(a_true, w_true, q_true);
        
        % Stack Data
        % Inputs: Accel, Gyro, Mag
        batch_inputs = [accel_meas, gyro_meas, mag_meas];
        % Targets: Pos, Vel, Quat
        batch_targets = [p_true, v_true, compact(q_true)];
        
        all_inputs = [all_inputs; batch_inputs];
        all_targets = [all_targets; batch_targets];
    end
end