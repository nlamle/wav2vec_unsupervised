import os
import json
import glob
import matplotlib.pyplot as plt

# Find ALL hydra log files
log_files = glob.glob('/root/wav2vec_unsupervised/**/hydra_train.log', recursive=True)
if not log_files:
    print('Could not find hydra_train.log. Has training started?')
    exit()

# Sort files by modification time (newest first)
log_files.sort(key=os.path.getmtime, reverse=True)
log_path = log_files[0]
print(f'Reading from newest log: {log_path}')

updates, d_losses, g_losses = [], [], []

# Parse the JSON logs
with open(log_path, 'r') as f:
    for line in f:
        if 'train_inner' in line and '{"epoch"' in line:
            try:
                data = json.loads(line.split(' - ')[1].strip())
                updates.append(int(data['num_updates']))
                d_losses.append(float(data['d_loss']))
                g_losses.append(float(data['g_loss']))
            except Exception as e:
                continue

print(f'Successfully parsed {len(updates)} training updates.')

if len(updates) == 0:
    print('Error: Found the log file, but it had no training data. Check if your wrapper script redirected the logs elsewhere.')
    exit()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(updates, d_losses, label='Discriminator Loss (d_loss)', color='red', alpha=0.8)
plt.plot(updates, g_losses, label='Generator Loss (g_loss)', color='blue', alpha=0.8)
plt.xlabel('Number of Updates')
plt.ylabel('Loss')
plt.title('Wav2Vec-U GAN Loss Trajectory (50k Run)')
plt.legend()
plt.grid(True)
plt.savefig('/root/wav2vec_unsupervised/loss_plot.png')
print('SUCCESS: Saved plot to /root/wav2vec_unsupervised/loss_plot.png')
