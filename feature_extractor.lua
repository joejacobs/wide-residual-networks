require 'math'
require 'nn'
local npy4th = require 'npy4th'
require 'torch'
local Utils = require 'Utils'

function get_options()

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Use a pre-saved neural network as a feature extractor')
    cmd:text()
    cmd:text('Options')
    cmd:option('-batch_size',1000,'minibatch size')
    cmd:option('-data_dir','nil','path to data')
    cmd:option('-inputs','RGB','input features')
    cmd:option('-model_path','nil','path to saved model')
    cmd:option('-n_del',1,'number of layers to remove from the model')
    cmd:option('-out_path','nil','path to output')
    cmd:option('-train_data','nil','path to training data (for mean and std)')

    opt = cmd:parse(arg)

    if opt.data_dir == 'nil' or opt.model_path == 'nil' or
        opt.out_path == 'nil' or opt.train_data == 'nil' then

        error('Data, model and output paths are all required')
    end

    opt.seed = 1234
end

-- launch cuda if necessary and if possible. fallback to CPU if not
function launch_cuda()

    io.write('Importing cuda...')
    local ok,ok2,ok3
    ok,cunn = pcall(require,'cunn')
    ok2,cudnn = pcall(require,'cudnn')
    ok3,cutorch = pcall(require,'cutorch')

    if not ok then

        print('package cudnn not found!')
    end

    if not ok2 then

        print('package cudnn not found!')
    end

    if not ok3 then

        print('package cutorch not found!')
    end

    if ok and ok2 and ok3 then

        print('successful!')
        cutorch.setDevice(1)
        cutorch.manualSeedAll(opt.seed)
        opt.cuda = true
    else

        print('Falling back to CPU mode')
        opt.cuda = false
    end
end

function main()

    get_options()
    launch_cuda()

    -- load trained nn
    local net = torch.load(opt.model_path)

    -- remove classification layers
    for i = 1,opt.n_del do

        net:remove()
    end

    if opt.cuda then

        cudnn.convert(net,cudnn)
    end

    -- load data
    print('Launching data loader')
    local data = Utils.data_to_th( npy4th.loadnpy( string.format(
        '%s/PreComp/Features/%s.npy',opt.data_dir,opt.inputs) ):double() )
    local n_batches = math.ceil( data:size(1) / opt.batch_size )

    -- get mean and std from train data
    local train_data = torch.load(opt.train_data)
    local mean,std = train_data.trainData.mean,train_data.trainData.std
    train_data = nil
    collectgarbage()

    -- normalize data
    data:add(-mean):div(std)

    -- get nn features for data, one minibatch at a time
    local fst,inputs,lst,nn_features,tmp = nil,nil,nil,nil,nil

    -- set network in evaluation mode
    net:evaluate()

    for i = 1,n_batches do

        -- get minibatch
        fst = ( (i - 1) * opt.batch_size ) + 1
        lst = math.min( i * opt.batch_size, data:size(1) )
        inputs = data[ { {fst,lst}, {}, {}, {} } ]

        -- convert to cuda
        if opt.cuda then

            inputs = inputs:cuda()
        end

        -- get network output
        tmp = net:forward(inputs):double()

        -- concatenate or store in output variable
        if nn_features then

            nn_features = torch.cat(nn_features,tmp,1)
        else
            
            nn_features = tmp
        end
    end

    -- save features to file
    Utils.save_to_hdf5(opt.out_path,'/data',nn_features)
end

main()
