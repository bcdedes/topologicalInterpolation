%% Generate Synthetic Dataset for Multi-Frame Super-Resolution

rng(453,'twister');
r = 3;
r2 = r^2 - 1;
noise_var = 25;

delta = [0 0; randi([-(r-1) (r-1)],r2,2)/r];
theta = [0; (-4 + (8)*rand(r2,1))];

% Print rotation and translation parameters to use in OpenCV where Point2f
% stores indices as (col, row), hence columns are printed first
disp( ['Theta: ', regexprep(num2str(theta',' %+.4f'),'\s+',', ')] );
disp( ['Delta: ', regexprep(num2str(delta(:,2)',' %+.4f'),'\s+',', '),...
newline, '       ', regexprep(num2str(delta(:,1)',' %+.4f'),'\s+',', ')] );

im = importdata('plate.tif');
s = synthDataGen(im, delta, theta, r, noise_var);
saveFrames('lowResolutionFrames', 'plate_3x', s)


function s = synthDataGen( im, delta, theta, scale, var )
%SYNTHDATAGEN Generate low resolution shifted and rotated images
%   S = synthDataGen(IM, DELTA, THETA, SCALE, VAR) generates low resolution 
%   frames from image IM using rotation parameters in THETA and translation 
%   parameters in DELTA 
% Inputs:
%   im    - Grayscale HR image (uint8)
%   delta - A matrix of two column vectors containing translation values
%           along x and y directions, respectively
%   theta - Column vector containing rotation values
%   scale - scaling (downsampling) factor
%   var   - noise variance (between 0 and 255)
%   snr   - (if specified) Signal-to-Noise Ratio after adding noise 

if length(theta) == length(delta); n = length(theta);
else; error("Number of shift and rotation parameters are not equal"); 
end

if ~(mod(scale, 1) == 0 && scale > 1)
  error("Scaling factor must be an integer and greater than 1!");
end

if ~isa(im, 'uint8')
  im = im2uint8( mat2gray(im) );
  warning('Input HR image is being converted to grayscale with intensity range 0-255');
end

s = cell(n, 1);
sz = size(im);

% make image size divisible by scaling factor
pad_r = scale - mod(sz(1), scale);
pad_c = scale - mod(sz(2), scale);
if (pad_r ~= scale)
  im = padarray(im, [pad_r 0], 'replicate', 'post');
  sz(1) = sz(1) + pad_r;
end
if (pad_c ~= scale)
  im = padarray(im, [0 pad_c], 'replicate', 'post');
  sz(2) = sz(2) + pad_c;
end

% warp
for i = 1:n
  s{i} = imtranslate(im, [-delta(i,2)*scale -delta(i,1)*scale], 'cubic');
  if (theta(i) ~= 0)
    s{i} = imrotate(s{i}, theta(i), 'bicubic', 'crop');
  end
end

% blur and downsample
for i = 1:n
  b = reshape(s{i}, scale, sz(1)/scale, scale, sz(2)/scale);
  s{i} = uint8(squeeze( sum(sum(b, 1), 3) / (scale*scale) )); % mean filter
end

if (nargin == 5 && var ~= 0)
  for i = 1:n
    s{i} = imnoise(s{i}, 'gaussian', 0, (sqrt(var)/255)^2);
  end
end

end


function saveFrames(folderPath, folderName, LR_frames)
% Save LR Frames to a folder
mkdir(folderPath, folderName);

for i = 1:length(LR_frames)
  fileName = strcat(folderPath, '/', folderName, '/', folderName, '_', num2str(i, '%02.f'), '.tif');
  imwrite(LR_frames{i}, fileName);
end

end
