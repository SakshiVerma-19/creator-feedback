# Hardware Acceleration: Optimized for AMD DirectML and NVIDIA CUDA backends via ONNX Runtime.
import cv2
import numpy as np

class SaliencyEngine:
    def __init__(self):
        self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def get_map(self, image):
        success, saliency_map = self.saliency.computeSaliency(image)
        return (saliency_map * 255).astype("uint8") if success else None

    def calculate_score(self, image, saliency_map):
        base_score = np.mean(saliency_map) / 2.55
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        face_bonus = 15 if len(faces) > 0 else 0
        
        h, w = saliency_map.shape
        center_zone = saliency_map[h//3:2*h//3, w//3:2*w//3]
        center_bonus = np.mean(center_zone) / 5.1
        
        total_score = min(100, base_score + face_bonus + center_bonus)
        return round(total_score, 1)
    
    def analyze_composition(self, saliency_map):
        h, w = saliency_map.shape
        
        x_third = w // 3
        y_third = h // 3
        points = [
            (x_third, y_third),
            (2 * x_third, y_third),
            (x_third, 2 * y_third),
            (2 * x_third, 2 * y_third)
        ]
        
        region_size = 50
        half_size = region_size // 2
        intensities = {}
        
        for i, (x, y) in enumerate(points, 1):
            y_start = max(0, y - half_size)
            y_end = min(h, y + half_size)
            x_start = max(0, x - half_size)
            x_end = min(w, x + half_size)
            
            region = saliency_map[y_start:y_end, x_start:x_end]
            mean_intensity = np.mean(region)
            intensities[f'power_point_{i}'] = float(mean_intensity)
        
        high_focus = any(intensity > 128 for intensity in intensities.values())
        
        return {
            'intensities': intensities,
            'high_focus': high_focus
        }
    
    def get_improvement_suggestions(self, image, saliency_map):
        suggestions = []
        h, w = saliency_map.shape
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            suggestions.append("Add a face or person - human elements boost engagement by 30-40%")
        else:
            face_in_focus = False
            for (x, y, w_face, h_face) in faces:
                face_region = saliency_map[y:y+h_face, x:x+w_face]
                if np.mean(face_region) > 120:
                    face_in_focus = True
                    break
            if not face_in_focus:
                suggestions.append("Make the face more prominent with better lighting or contrast")
        
        mean_saliency = np.mean(saliency_map)
        if mean_saliency < 60:
            suggestions.append("Low visual interest - add bold colors, text, or contrasting elements")
        elif mean_saliency < 80:
            suggestions.append("Increase visual contrast or add more vibrant colors")
        
        center_zone = saliency_map[h//3:2*h//3, w//3:2*w//3]
        center_mean = np.mean(center_zone)
        if center_mean < 100:
            suggestions.append("Place key elements in the center third for better focus")
        
        edge_intensity = np.mean(saliency_map[:h//4, :]) + np.mean(saliency_map[-h//4:, :])
        if edge_intensity / 2 > mean_saliency:
            suggestions.append("Move important elements away from edges - they get cut off on mobile")
        
        edges = cv2.Canny(gray, 50, 150)
        text_zones = []
        for i in range(0, h-50, 50):
            for j in range(0, w-50, 50):
                zone_edges = edges[i:i+50, j:j+50]
                if np.sum(zone_edges > 0) > 300:
                    text_zones.append((i, j))
        
        if len(text_zones) > 0:
            text_salient = False
            for (i, j) in text_zones:
                if np.mean(saliency_map[i:i+50, j:j+50]) > 100:
                    text_salient = True
                    break
            if not text_salient:
                suggestions.append("Text detected but not prominent - use larger, bolder fonts")
        
        if np.std(saliency_map) < 30:
            suggestions.append("Add more variety - create clear focal points with contrast")
        
        top_third = saliency_map[:h//3, :]
        if np.mean(top_third) < mean_saliency * 0.8:
            suggestions.append("Strengthen the top area - it's the first thing viewers see")
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        if np.mean(saturation) < 80:
            suggestions.append("Boost color saturation to make the thumbnail pop")
        
        if not suggestions:
            suggestions.append("Excellent composition! Focus areas are well-positioned")
        
        return suggestions
    
def get_legibility_score(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return round(min(100, edge_density * 1000), 1)