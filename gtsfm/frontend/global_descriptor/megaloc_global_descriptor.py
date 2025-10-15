# gtsfm/frontend/global_descriptor/megaloc_global_descriptor.py
from thirdparty.megaloc.megaloc import MegaLocModel

class MegaLocGlobalDescriptor(GlobalDescriptorBase):
    def __init__(self) -> None:
        logger.info("⏳ Loading MegaLoc model weights...")
        self._model = MegaLocModel().eval()
        
    def describe(self, image: Image) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        
        img_array = image.value_array.copy()
        img_tensor = (
            torch.from_numpy(img_array)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .type(torch.float32) / 255.0
        )
        
        with torch.no_grad():
            descriptor = self._model(img_tensor)
            
        return descriptor.detach().squeeze().cpu().numpy()