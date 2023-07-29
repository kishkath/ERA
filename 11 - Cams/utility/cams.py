from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import torchvision.models as models
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from visualizers import mis_prediction
import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


storing_images, storing_predicted_labels, storing_target_labels = mis_prediction()

def plotting_gradcam(pil_img):
    resnet = models.resnet18(pretrained=True)
    torch_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])(pil_img).to(device)

    normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
    configs = [
        dict(model_type='resnet', arch=resnet, layer_name='layer2')]

    for config in configs:
        config['arch'].to(device).eval()

    cams = [[cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
            for config in configs]

    images = []
    for gradcam, gradcam_pp in cams:
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])

    grid_image = make_grid(images, nrow=6)
    return transforms.ToPILImage()(grid_image)

def plotting_gradCams(imagesneeded):
    print("Diagnosis is happening for Layer2 of ResNet18. Lets go!")
    figure2 = plt.figure(figsize=(16, 32))
    for i in range(imagesneeded):
        sub = figure2.add_subplot(imagesneeded,1, i + 1)
        p = plotting_gradcam(transforms.ToPILImage()(storing_images[i]))
        sub.imshow(p)
        sub.set_title(
            f"Predicted as: {classes[storing_predicted_labels[i]]} \n But, Actual is: {classes[storing_target_labels[i]]}")
    plt.tight_layout()
    plt.savefig("./images/gradCAM.jpg")
