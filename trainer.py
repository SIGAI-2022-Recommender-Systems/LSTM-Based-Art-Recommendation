import torch
from tqdm import tqdm
import pandas as pd
from data_loader import *
from model import *
from torch.optim import Adam
def BPRLoss(output:torch.Tensor,target:torch.Tensor,negatives:torch.Tensor):
    # print(target.size(),output.size(),negatives.size())
    assert target.size()[0] == negatives.size()[0] == output.size()[0]
    assert target.size()[1] == negatives.size()[1] == output.size()[1]
    # [:,1:-1] removes the predictions for 0th time step and last time step since they are meaningless for loss
    # torch.gather grabs the positive and negative at each time step
    return - (torch.gather(output[:,1:-1,:],2,target[:,1:-1].unsqueeze(2)) - torch.gather(output[:,1:-1,:],2,negatives[:,1:-1].unsqueeze(2))).sigmoid().log().sum()

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model:torch.nn.Module, criterion, metric_ftns, optimizer:torch.optim.Adam,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device="cuda"
        self.data_loader = data_loader
        self.log = pd.DataFrame(columns = ["Epoch"]+metric_ftns)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        losses = torch.zeros(len(self.data_loader))
        pbar = tqdm(enumerate(self.data_loader),total=len(self.data_loader),delay=.5)
        for batch_idx, (data,counts) in pbar:
            target = torch.roll(data, -1, 1).to(self.device)
            self.optimizer.zero_grad()
            #generate negative items
            negatives = target.clone()
            while len(negatives[negatives==target]):
                negatives[negatives==target] = torch.randint(2,1569973,size=negatives[negatives==target].size(),device=self.device)
            output = self.model(data)
            #output format: userxtimestepxprediction
            loss = self.criterion(output, target,negatives)
            loss.backward()
            self.optimizer.step()
            losses[batch_idx]= loss
        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     log.update(**{'val_'+k : v for k, v in val_log.items()})

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
            pbar.set_description(f"Batch loss:{str(losses.mean().item())}")
        return losses.mean()

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data) in tqdm(enumerate(self.valid_data_loader,total=len())):
                target = torch.roll(data, -1, 1)
                #generate negative items
                negatives = data[:].to(self.device)
                while len(negatives[negatives==data]):
                    negatives[negatives==data] = torch.randint(2,1569973,size=negatives[negatives==data].size())
                output = self.model(data)
                loss = self.criterion(output[:-1], target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
    
    def train(self,epochs=30):
        for i in range(epochs):
            loss = self._train_epoch(i)
            tqdm.write(f"Epoch {i} loss: {str(loss)}")
    
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        
if __name__ == "__main__":
    dataset = recSysDataset(max_len=20, root = "data\\kcore_5_collated.txt")
    train_data = train_val_test_split(dataset, split=(.5,.45,.05), mode = "train")
    val_data = train_val_test_split(dataset, split=(.5,.45,.05), mode = "val")
    test_data = train_val_test_split(dataset, split=(.5,.45,.05), mode = "test")

    dataloader = torch.utils.data.DataLoader(train_data, batch_size = 16, shuffle = True)
    tqdm.write("Dataloader done")
    model = recSysNet("cpu",1569974,20,20,1,.1,bidirectional=False)
    tqdm.write("Model instantiated")
    trainer = Trainer(model, BPRLoss, [], Adam(model.parameters(),lr=.001), dataloader)
    tqdm.write("Training...")
    trainer.train()