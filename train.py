# train.py - QuartzNet Training Pipeline
import yaml, torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from models import QuartzNet

class QuartzNetTrainer:
    def __init__(self, model: QuartzNet, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=float(config['training'].get('learning_rate',1e-3)))
        self.writer = SummaryWriter(config['training'].get('log_dir','logs'))

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
            mel_spec, targets, target_lengths = batch['mel_spec'].to(self.device), batch['text'].to(self.device), batch['target_lengths'].to(self.device)
            output = self.model(mel_spec)
            output_lengths = torch.full((mel_spec.size(0),), output.size(2), dtype=torch.long, device=self.device)
            loss = self.criterion(output.permute(2,0,1), targets, output_lengths, target_lengths)
            self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0: self.writer.add_scalar('Loss/train', loss.item(), epoch*len(dataloader)+batch_idx)
        return total_loss / max(1,len(dataloader))

    def save_checkpoint(self, epoch: int, loss: float):
        ckpt_dir = Path(self.config['training'].get('checkpoint_dir','checkpoints'))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"quartznet_epoch_{epoch}.pt"
        torch.save({'epoch':epoch,'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict(),'loss':loss}, path)
        print(f"Checkpoint saved: {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config.yml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config,"r",encoding="utf-8"))
    model = QuartzNet(); trainer = QuartzNetTrainer(model, cfg)
    print(f"Training on device: {trainer.device}")
    # add your DataLoader loop here
