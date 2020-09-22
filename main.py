from PIL import Image
import numpy as np

def cinza(imgCol):
    w, h = imgCol.size
    img = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            pxl = imgCol.getpixel((x,y))
            # media das coordenadas RGB
            lum = (pxl[0] + pxl[1] + pxl[2])//3
            img.putpixel((x,y), (lum, lum, lum))
    return img

def cinza_ponderada(imgCol):
    w, h = imgCol.size
    img = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            pxl = imgCol.getpixel((x,y))
            # media ponderada das coordenadas RGB
            lum = int(0.299*pxl[0] + 0.587*pxl[1] + 0.114*pxl[2])
            img.putpixel((x,y), (lum, lum, lum))
    return img

def limiarizacao(imgCol):
    w, h = imgCol.size
    img = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            pxl = imgCol.getpixel((x, y))
            
            media = (pxl[0] + pxl[1] + pxl[2]) / 3

            calc = 255 if media > 127.5 else 0

            img.putpixel((x, y), (calc, calc, calc))

    return img

def negativa(imgCol):
    w, h = imgCol.size
    img = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            pxl = imgCol.getpixel((x, y))

            img.putpixel((x, y), (255 - pxl[0], 255 - pxl[1], 255 - pxl[2]))

    return img

def soma(imgCol1, imgCol2, val1 = 1, val2 = 1):
    val1 = 1 if val1 > 1.0 else val1
    val2 = 1 if val2 > 1.0 else val2

    w1, h1 = imgCol1.size
    w2, h2 = imgCol2.size
    
    menW = w1 if w1 < w2 else w2
    menH = h1 if h1 < h2 else h2

    img = Image.new("RGB", (menW, menH))

    for x in range(menW):
        for y in range(menH):
            pxlImg1 = imgCol1.getpixel((x, y))
            pxlImg2 = imgCol2.getpixel((x, y))

            img.putpixel((x, y), (int((pxlImg1[0]*val1 + pxlImg2[0]*val2) / 2), int((pxlImg1[1]*val1 + pxlImg2[1]*val2) / 2), int((pxlImg1[2]*val1 + pxlImg2[2]*val2) / 2)))

    return img

def subtracao(imgCol1, imgCol2):
    w1, h1 = imgCol1.size
    w2, h2 = imgCol2.size
    
    menW = w1 if w1 < w2 else w2
    menH = h1 if h1 < h2 else h2

    img = Image.new("RGB", (menW, menH))

    for x in range(menW):
        for y in range(menH):
            pxlImg1 = imgCol1.getpixel((x, y))
            pxlImg2 = imgCol2.getpixel((x, y))

            red = pxlImg1[0] - pxlImg2[0]
            green = pxlImg1[1] - pxlImg2[1]
            blue = pxlImg1[2] - pxlImg2[2]

            red = 0 if red < 0 else red
            green = 0 if green < 0 else green
            blue = 0 if blue < 0 else blue

            img.putpixel((x, y), (red, green, blue))

    return img

def pontos_salientes(imgCol):
    w, h = imgCol.size
    img = novaImagem(w, h)

    mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    x = 1
    y = 1

    for x in range(w - 1):
        for y in range(h - 1):
            matPxl = matrizPixels(imgCol, x, y)

            matResult = multMatrizes(matPxl, mask)
            soma = somaValMatriz(matResult)

            img.putpixel((x, y), (int(soma), int(soma), int(soma)))

    return img

def dilatacao(imgCol):
    w, h = imgCol.size
    img = novaImagem(w, h)

    mask = np.zeros((5, 5))
    hotspot = (2, 2)

    for j in range(4):
        mask[2][j] = 1

    x, y = 2

    for x in range(w - 2):
        for y in range(h - 2):
            pxl = imgCol.getpixel((x, y))

            if int(media_pixel(pxl)) == 255:
                matPxl = matrizPixels(imgCol, x, y)
                img = aplica_mask_dilatacao(img, mask, x, y)


    return img

def media_pixel(pixel):
    return (pixel[0] + pixel[1] + pixel[2]) / 3

def novaImagem(w, h):
    img = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            img.putpixel((x, y), (0, 0, 0))

    return img  

def matrizPixels(imgCol, x, y):
    tam = mat.shape[0]

    matResult = np.zeros((tam, tam))

    lim = (tam - 1) / 2

    i, j = -lim 
    
    while i <= lim:
        while j <= lim:
            pxl = imgCol.getpixel((x + i, y + j))
            corResult = (pxl[0] + pxl[1] + pxl[2]) / 3

            matResult[x + i][y + j] = int(corResult)

    return matResult

def multMatrizes(mat1, mat2):
    tam = mat1.shape[0]

    matResult = np.zeros((tam, tam))

    for x in range(tam - 1):
        for y in range(tam - 1):         
            matResult[x][y] = mat1[x][y] * mat2[x][y]

    return matResult

def somaValMatriz(mat):
    soma = 0
    tam = mat.shape[0]

    for x in range(tam - 1):
        for y in range(tam - 1):
            soma += mat[x][y]

    return soma

def aplica_mask_dilatacao(imgNova, mask, x, y):
    tam = mask.shape[0]

    lim = (tam - 1) / 2

    i, j = -lim 
    
    while j <= lim:
        while i <= lim:
            if mask[i][j] == 1:
                imgNova.putpixel((x + i, y + j), (255, 255, 255))

    return imgNova

if __name__ == "__main__":
    # Cinza
    #img = Image.open("imgsOriginais\\outono.jpg")
    #cinza(img).save("imgsModificadas\\paisagemCinza.jpg")

    # Limiarizacao
    #img2 = Image.open("imgsOriginais\\castelo.jpg")
    #limiarizacao(img2).save("imgsModificadas\\paisagem2Limiarizada.jpg")

    # Negativo
    #img3 = Image.open("imgsOriginais\\rioDeJaneiro.jpg")
    #negativa(img3).save("imgsModificadas\\paisagem3Negativa.jpg")

    # Soma e soma ponderada de imagens
    #img4 = Image.open("imgsOriginais\\rioDeJaneiro.jpg")
    #img5 = Image.open("imgsOriginais\\castelo.jpg")
    #soma(img4, img5, val1=0.1).save("imgsModificadas\\imagensSomadas.jpg")

    # Subtracao de imagens
    #img6 = Image.open("imgsOriginais\\ferrari1.jpg")
    #img7 = Image.open("imgsOriginais\\ferrari2.jpg")
    #subtracao(img6, img7).save("imgsModificadas\\imagensSubtraidas.jpg")

    # Pontos salientes
    #img8 = Image.open("imgsOriginais\\oncaPintada.jpg")
    #pontos_salientes(img8).save("imgsModificadas\\oncaPintadaSaliente.jpg")

    # Dilatacao
    img9 = Image.open("imgsOriginais\\emojiFeliz.png")
    dilatacao(limiarizacao(img8)).save("imgsModificadas\\emojiDilatado.png")