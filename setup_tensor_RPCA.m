%% set paths
addpath(genpath('.'))

%% downlaod toolboxes

set(0,'DefaultFigureVisible','off')

prompt = 'About to download escalator data. Continue (y/n)?';
str = input(prompt,'s');
if isempty(str)
    str = 'Y';
end

if strcmp(str,'y')
    downloadEscalatorData
else
    disp('Skipping download.')
end

prompt = 'About to download tensor_toolbox from GitLab. Continue (y/n)?';
str = input(prompt,'s');
if isempty(str)
    str = 'Y';
end

if strcmp(str,'y')
    try 
        system('git clone https://gitlab.com/tensors/tensor_toolbox.git util/tensor_toolbox/');
    catch
        error('Download of tensor_toolbox failed. Do you already have it in util/?')
    end
else
    disp('Skipping download.')
end

prompt = 'About to download and compile topic modeling dependencies from GitHub. Continue (y/n)?';
str = input(prompt,'s');
if isempty(str)
    str = 'Y';
end

if strcmp(str,'y')
    try 
%         system('cd TensorTopicModeling/dependency/');
        cd TensorTopicModeling/dependency/
        system('git clone https://github.com/FurongHuang/TensorDecomposition4TopicModeling.git');
        cd TensorDecomposition4TopicModeling/TopicModelingSingleNodeALS/TopicModel/TopicModel/
        system('rm -r exe*');
        system('make');
        cd ../../../../../
        system('make');
        cd ../
    catch
        error('Download of TopicModelingSingleNodeALS failed.')
    end
else
    disp('Skipping download.')
end

%% Add newly downloaded paths
addpath(genpath('.'))
