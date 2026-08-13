import matplotlib.pyplot as plt
import torch


def plot(x: torch.Tensor, idx = 0, name:str = None, vmin: float = 0.0, vmax: float = 1.0):
    if name is None:
        plt.imshow(x[:, 0 + idx : 1 + idx , ...].detach().cpu()[0, 0, ...], cmap='gray', vmin=vmin, vmax=vmax)
        plt.show()
    else:
        plt.imsave(name, x[:, 0 + idx : 1 + idx , ...].detach().cpu()[0, 0, ...], cmap='gray', vmin=vmin, vmax=vmax)


def plot_3d(tensor):
        """
        Plots middle slices of a 5D tensor (1, C, D, H, W) for each channel.
        
        For each channel, it shows:
            - middle slice along depth (D)
            - middle slice along height (H)
            - middle slice along width (W)
        
        Args:
            tensor (torch.Tensor): Input tensor of shape (1, C, D, H, W)
        """
        # assert tensor.ndim == 5 and tensor.shape[0] == 1, \
        #     f"Expected tensor of shape (1, C, D, H, W), got {tensor.shape}"
        
        tensor = tensor.squeeze(0)  # → shape: (C, D, H, W)
        C, D, H, W = tensor.shape
        
        d_mid = D // 2
        h_mid = H // 2
        w_mid = W // 2
        
        fig, axes = plt.subplots(C, 3, figsize=(9, 3*C))
        
        if C == 1:
            axes = axes[None, :]  # ensure 2D array of axes even if one channel
        
        for c in range(C):
            # Middle slice along each axis
            slice_d = tensor[c, d_mid, :, :].detach().cpu()
            slice_h = tensor[c, :, h_mid, :].detach().cpu()
            slice_w = tensor[c, :, :, w_mid].detach().cpu()
            
            axes[c, 0].imshow(slice_d, cmap='gray', vmax=0.048)
            axes[c, 0].set_title(f'Channel {c}: mid-D')
            
            axes[c, 1].imshow(slice_h, cmap='gray', vmax=0.048)
            axes[c, 1].set_title(f'Channel {c}: mid-H')
            
            axes[c, 2].imshow(slice_w, cmap='gray', vmax=0.048)
            axes[c, 2].set_title(f'Channel {c}: mid-W')
            
            for ax in axes[c]:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()