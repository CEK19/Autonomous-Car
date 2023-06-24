from PIL import Image, ImageEnhance

img = Image.open('../assets/demo/yellow1.jpg')

# tăng giảm độ choá
def adjust_contrast(image, factor):
	enhancer = ImageEnhance.Contrast(image)
	return enhancer.enhance(factor)

# tăng giảm độ sáng
def adjust_brightness(image, factor):
	enhancer = ImageEnhance.Brightness(image)
	return enhancer.enhance(factor)
