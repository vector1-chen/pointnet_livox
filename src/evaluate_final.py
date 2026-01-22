import torch
from torch.utils.data import DataLoader
from models.pointnet import PointNetSegmentation
from data.dataset import S3DISDataset
from utils.metrics import evaluate_model
from utils.visualization import visualize_multiple_samples, visualize_predictions, class_names

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Configuration
# num_classes = 13
# num_points = 4096
# test_area = 5
# batch_size = 16
# data_dir = './s3dis_data/processed'

# # Load model
# print("Loading best model...")
# model = PointNetSegmentation(num_classes=num_classes, feature_transform=True).to(device)
# model.load_state_dict(torch.load('checkpoints/best_pointnet_s3dis.pth', map_location=device))
# model.eval()

# # Load test dataset
# print("Loading test dataset...")
# test_dataset = S3DISDataset(data_dir, num_points=num_points, split='test', test_area=test_area)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# print(f"Test dataset size: {len(test_dataset)} samples")

# # Evaluate on test set
# print("\n" + "="*60)
# print("FINAL EVALUATION")
# print("="*60)

# mean_iou, class_ious = evaluate_model(model, test_loader, device, class_names)

# print(f"\nEvaluation completed! Best mIoU: {mean_iou:.4f}")

# Visualize predictions
print("\nVisualizing model predictions...")
print("="*50)

#visualize_predictions(num_samples=3)
visualize_multiple_samples(num_samples=5)

print("\n" + "="*60)
print("Evaluation and visualization completed!")
print("="*60)
