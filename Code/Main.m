clear; clc;
close all
tstart = datetime;

%% ================== GLOBAL SETTINGS ==================
MAX_EPISODES          = 3500;
MODELS_PATH           = 'results/';
VALVE_SIMULATION_MODEL = 'Train';
AGENT_BLOCK           = [VALVE_SIMULATION_MODEL '/RL Sub-System/RL Agent'];
VERSION               = 'RL_Grade_I';
USE_PRE_TRAINED_MODEL = false;
PRE_TRAINED_MODEL_FILE = 'RL_Grade_I.mat';

if ~isfolder(MODELS_PATH), mkdir(MODELS_PATH); end

%% ================== PLANT / ESO / DISCRETIZATION ==================
plant.Ts   = 1e-3;
plant.L    = 0.4;
plant.m    = round(plant.L/plant.Ts);    % delay steps
plant.TIME_DELAY = plant.L;

plant.fS   = 8.4/5;
plant.fD   = 3.5243/5;

plant.K    = 3.8163;
plant.tau  = 156.46;

a          = exp(-plant.Ts/plant.tau);
b          = plant.K*(1 - a);
g          = b * 1.7;

plant.a    = a;
plant.b    = b;
plant.g    = g;

% ESO (simple Euler discretization of continuous gains)
w0         = 0.5;
beta1_c    = 2*w0;
beta2_c    = w0^2;

eso.beta1  = beta1_c*plant.Ts;
eso.beta2  = beta2_c*plant.Ts;

%% ================== CONTROLLER / SCALING PARAMETERS ==================
ctrl.UMAX  = 10;
ctrl.kp    = 0.20;
ctrl.krl   = 0.30*ctrl.UMAX;

scale.Emax = 30;
scale.Ymax = 200;
scale.Dmax = g*ctrl.UMAX;
scale.Umax = ctrl.UMAX;

integrator.Tf_int = 50;

% (these are defined but unused in original code - kept for completeness)
weights.q   = 1;
weights.rho = 0.01;
weights.eta = 0.02;
weights.w_i = 0.25;

%% ================== EXPORT TO BASE WORKSPACE ==================
exportToBase(plant);
exportToBase(eso);
exportToBase(ctrl);
exportToBase(scale);
exportToBase(integrator);
exportToBase(weights);

%% ================== RL ENVIRONMENT SPECS ==================
nObs = 4;
nAct = 1;

obsInfo = rlNumericSpec([nObs 1], ...
    'LowerLimit', -inf(nObs,1), ...
    'UpperLimit',  inf(nObs,1));
obsInfo.Name        = 'observations';
obsInfo.Description = 'e, y, e_int, u0_prev';

actInfo = rlNumericSpec([nAct 1], ...
    'LowerLimit', -1, ...
    'UpperLimit',  1);
actInfo.Name = 'u0_raw';

env = rlSimulinkEnv(VALVE_SIMULATION_MODEL, AGENT_BLOCK, obsInfo, actInfo);
env.ResetFcn = @(in)localResetFcn(in);

%% ================== DDPG HYPERPARAMETERS ==================
hyper.criticLearningRate = 1e-3;
hyper.actorLearningRate  = 1e-4;
hyper.GAMMA              = 0.95;
hyper.BATCH_SIZE         = 256;
hyper.BUFFER_LEN         = 5e5;
hyper.TargetSmooth       = 1e-3;

hyper.neuronsCriticFC1   = 64;
hyper.neuronsCriticFC2   = 64;
hyper.neuronsActorFC1    = 64;
hyper.neuronsActorFC2    = 64;

%% ================== CRITIC (Q(s,a)) ==================
criticNet = createCriticNetwork(nObs, nAct, ...
    hyper.neuronsCriticFC1, hyper.neuronsCriticFC2, ...
    hyper.neuronsActorFC1);

criticOpts = rlRepresentationOptions( ...
    'LearnRate',         hyper.criticLearningRate, ...
    'GradientThreshold', 1, ...
    'UseDevice',         'gpu');

critic = rlQValueRepresentation(criticNet, obsInfo, actInfo, ...
    'ObservationInputNames', {'State'}, ...
    'ActionInputNames',      {'Action'}, ...
    criticOpts);

%% ================== ACTOR (mu(s)) ==================
actorNet = createActorNetwork(nObs, nAct, ...
    hyper.neuronsActorFC1, hyper.neuronsActorFC2);

actorOpts = rlRepresentationOptions( ...
    'LearnRate',         hyper.actorLearningRate, ...
    'GradientThreshold', 1, ...
    'UseDevice',         'gpu');

actor = rlDeterministicActorRepresentation(actorNet, obsInfo, actInfo, ...
    'ObservationInputNames', {'State'}, ...
    actorOpts);

%% ================== DDPG AGENT ==================
agentOpts = rlDDPGAgentOptions( ...
    'SampleTime',                    plant.Ts, ...
    'TargetSmoothFactor',            hyper.TargetSmooth, ...
    'DiscountFactor',                hyper.GAMMA, ...
    'MiniBatchSize',                 hyper.BATCH_SIZE, ...
    'ExperienceBufferLength',        hyper.BUFFER_LEN, ...
    'ResetExperienceBufferBeforeTraining', true, ...
    'SaveExperienceBufferWithAgent', false);

% Exploration noise
agentOpts.NoiseOptions.Variance          = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;

if USE_PRE_TRAINED_MODEL
    tmp   = load(fullfile(MODELS_PATH, PRE_TRAINED_MODEL_FILE), 'agent');
    agent = tmp.agent;
else
    agent = rlDDPGAgent(actor, critic, agentOpts);
end

%% ================== TRAINING OPTIONS ==================
Tf        = 60;
maxsteps  = ceil(Tf/plant.Ts);
AVG_WIN   = 50;

trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',                 MAX_EPISODES, ...
    'MaxStepsPerEpisode',          maxsteps, ...
    'ScoreAveragingWindowLength',  AVG_WIN, ...
    'Verbose',                     false, ...
    'Plots',                       'training-progress', ...
    'UseParallel',                 true, ...
    'StopTrainingCriteria',        'AverageReward', ...
    'StopTrainingValue',           1000);

%% ================== TRAIN ==================
fprintf('\n==== Training RL-ADRC ====\n');
fprintf('Model: %s | Ts=%.3f s | Delay=%.2f s (m=%d)\n', ...
    VALVE_SIMULATION_MODEL, plant.Ts, plant.L, plant.m);
fprintf('ESO: w0=%.3f rad/s | beta1=%.4f | beta2=%.6f | g=%.6f\n', ...
    w0, eso.beta1, eso.beta2, plant.g);

save_system(VALVE_SIMULATION_MODEL);
trainingStats = train(agent, env, trainOpts); %#ok<NASGU>

save(fullfile(MODELS_PATH, [VERSION '.mat']), "agent");

tend = datetime;
fprintf('\n✅ Training complete. Elapsed time: %s\n', string(tend - tstart));

%% ================== LOCAL FUNCTIONS ==================
function exportToBase(s)
    fn = fieldnames(s);
    for k = 1:numel(fn)
        assignin('base', fn{k}, s.(fn{k}));
    end
end

function criticNet = createCriticNetwork(nObs, nAct, nC1, nC2, nActFC)
    statePath = [
        featureInputLayer(nObs, 'Normalization','none', 'Name','State')
        fullyConnectedLayer(nC1, 'Name','CriticStateFC1')
        reluLayer('Name','CriticRelu1')
        fullyConnectedLayer(nC2, 'Name','CriticStateFC2')];

    actionPath = [
        featureInputLayer(nAct, 'Normalization','none', 'Name','Action')
        fullyConnectedLayer(nActFC, 'Name','CriticActionFC1')];

    commonPath = [
        additionLayer(2,'Name','add')
        reluLayer('Name','CriticCommonRelu')
        fullyConnectedLayer(1,'Name','CriticOutput')];

    criticNet = layerGraph();
    criticNet = addLayers(criticNet, statePath);
    criticNet = addLayers(criticNet, actionPath);
    criticNet = addLayers(criticNet, commonPath);
    criticNet = connectLayers(criticNet,'CriticStateFC2','add/in1');
    criticNet = connectLayers(criticNet,'CriticActionFC1','add/in2');
end

function actorNet = createActorNetwork(nObs, nAct, nA1, nA2)
    actorNet = [
        featureInputLayer(nObs, 'Normalization','none', 'Name','State')
        fullyConnectedLayer(nA1, 'Name','actorFC1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(nA2, 'Name','actorFC2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(nAct, 'Name','ActionPreTanh')
        tanhLayer('Name','Action')];
end

function in = localResetFcn(in)
    % Reference close to your y-scale
    REF_INIT = 5  + 25*rand();   % 5..30
    Y0       = 0  +  5*rand();   % small initial output
    WAMP     = 0;
    XKP0     = 0;

    in = setVariable(in,'REF_INIT',REF_INIT);
    in = setVariable(in,'XKP0',   XKP0);
    in = setVariable(in,'Y0',     Y0);
    in = setVariable(in,'WAMP',   WAMP);
end
