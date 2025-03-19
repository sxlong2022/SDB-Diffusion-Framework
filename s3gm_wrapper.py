def adaptive_transform_sampling(self, x):
    """采样时的自适应数据转换函数"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(self.device)
    
    # 获取掩码（第4个通道）
    mask = input_tensor[0, -1, 3:4]
    land_mask = torch.abs(x - self.config.land_value) < 0.1
    sea_mask = ~land_mask
    
    if sea_mask.any():
        sea_data = x[sea_mask]
        
        # 使用分位数统计而不是均值和标准差
        q25 = torch.quantile(sea_data, 0.25)
        q75 = torch.quantile(sea_data, 0.75)
        iqr = q75 - q25
        median = torch.median(sea_data)
        
        # 计算空间相关性权重
        H, W = x.shape[-2:]
        y_coords = torch.linspace(0, 1, H, device=self.device)
        x_coords = torch.linspace(0, 1, W, device=self.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([y_grid, x_grid], dim=-1)
        
        # 计算每个点与观测点的距离权重
        obs_points = (mask > 0).nonzero(as_tuple=True)
        if len(obs_points[0]) > 0:
            obs_coords = torch.stack([
                obs_points[0].float() / (H-1),
                obs_points[1].float() / (W-1)
            ], dim=-1)
            
            # 使用改进的空间权重计算
            coords_flat = coords.reshape(-1, 2)
            distances = torch.cdist(coords_flat, obs_coords)
            spatial_weights = torch.exp(-self.config.spatial_decay * distances)
            confidence = torch.exp(-distances.min(dim=1)[0]).reshape(H, W)
            
            # 根据置信度调整阈值
            threshold_factor = torch.lerp(
                torch.tensor(1.5, device=self.device),
                torch.tensor(3.0, device=self.device),
                confidence
            )
            
            # 应用自适应阈值
            curr_min = median - threshold_factor * iqr
            curr_max = median + threshold_factor * iqr
            
            # 渐进式缩放
            scale_factor = torch.lerp(
                torch.tensor(0.9, device=self.device),
                torch.tensor(1.0, device=self.device),
                confidence
            )
            
            x = torch.where(land_mask, x,
                          torch.where(x < curr_min,
                                    curr_min + (x - curr_min) * scale_factor,
                                    torch.where(x > curr_max,
                                              curr_max + (x - curr_max) * scale_factor,
                                              x)))
    
    # 处理NaN和无穷值
    x = torch.where(land_mask, x, torch.nan_to_num(x, nan=0.0))
    return x 