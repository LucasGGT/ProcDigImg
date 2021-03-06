from PIL import Image
import numpy as np
import math
import os

def limpar():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_horizontal(im1, im2):
    im = Image.fromarray(np.hstack((np.array(im1), np.array(im2))))
    im.show()
    return im

def show_vertical(im1, im2):
    im = Image.fromarray(np.vstack((np.array(im1), np.array(im2))))
    im.show()
    return im

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
            matPxl = matrizPixels(imgCol, x, y, 3)

            matResult = multMatrizes(matPxl, mask)
            soma = somaValMatriz(matResult)

            img.putpixel((x, y), (int(soma), int(soma), int(soma)))

    return img

def dilatacao(imgCol, mask = [[-1]]):
    w, h = imgCol.size
    img = novaImagem(w, h)
    tam = 0

    if mask[0][0] == -1:
        tam = 5

        mask = np.zeros((tam, tam))
        
        for lin in range(tam):
            for col in range(tam):
                mask[lin][col] = 1
    else:
        tam = mask.shape[0]

    lim = int((tam - 1) / 2)

    x = lim
    y = lim

    for x in range(w - lim):
        for y in range(h - lim):
            pxl = imgCol.getpixel((x, y))
            #matResulRep = matPxlRep(pxl, 5)

            if int(media_pixel(pxl)) != 0:
                i2 = -lim
                j2 = -lim
                for i in range(tam):
                    for j in range(tam):
                        if mask[i][j] == 1:
                            img.putpixel((x + i2, y + j2), pxl)
                        j2 += 1
                    j2 = -lim
                    i2 += 1

    return img

def erosao(imgCol, mask = [[-1]]):
    w, h = imgCol.size
    img = novaImagem(w, h)
    tam = 0
    
    if mask[0][0] == -1:
        tam = 9
        matrizFull = True

        mask = np.zeros((tam, tam))
        
        if matrizFull:
            for lin in range(tam):
                for col in range(tam):
                    mask[lin][col] = 1
        else:
            for j in range(tam):
                mask[int(tam / 2)][j] = 1
    else:
        tam = mask.shape[0]

    lim = int((tam - 1) / 2)

    x = lim
    y = lim

    for x in range(w - lim):
        for y in range(h - lim):
            if todosDifZero(imgCol, mask, x, y):
                i2 = -lim
                j2 = -lim
                for i in range(tam):
                    for j in range(tam):
                        if i != j:
                            img.putpixel((x + i2, y + j2), (0, 0, 0))
                        else:
                            img.putpixel((x + i2, y + j2), (255, 255, 255))
                        j2 += 1
                    j2 = -lim
                    i2 += 1

    return img

def abertura(imgCol):
    tam = 3
    matrizFull = False

    mask = np.zeros((tam, tam))
    
    if matrizFull:
        for lin in range(tam):
            for col in range(tam):
                mask[lin][col] = 1
    else:
        for j in range(tam):
            mask[int(tam / 2)][j] = 1
    
    imgErosao = erosao(imgCol, mask)

    return dilatacao(imgErosao, mask)

def fechamento(imgCol):
    tam = 5
    matrizFull = True

    mask = np.zeros((tam, tam))
    
    if matrizFull:
        for lin in range(tam):
            for col in range(tam):
                mask[lin][col] = 1
    else:
        for j in range(tam):
            mask[int(tam / 2)][j] = 1
    
    imgDilatacao = dilatacao(imgCol, mask)

    return erosao(imgDilatacao, mask)

def roberts(imgCol, valLim = 125):
    w, h = imgCol.size
    img = novaImagem(w, h)

    mask1 = np.array([[0, 1], [-1, 0]])
    mask2 = np.array([[1, 0], [0, -1]])

    for x in range(w - 1):
        for y in range(h - 1):
            if calcMaskRoberts(imgCol, x, y, mask1, mask2) > valLim:
                img.putpixel((x, y), (255, 255, 255))

    return img

def sobel(imgCol, valLim = 125):
    w, h = imgCol.size
    img = novaImagem(w, h)
    tam = 3
    lim = int((tam - 1) / 2)

    mask1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    mask2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    x = lim
    y = lim

    for x in range(w - lim):
        for y in range(h - lim):
            if calcMaskSobel(imgCol, x, y, mask1, mask2) > valLim:
                img.putpixel((x, y), (255, 255, 255))

    return img

def robinson(imgCol, valLim = 125):
    w, h = imgCol.size
    img = novaImagem(w, h)
    tam = 3
    lim = int((tam - 1) / 2)

    mask1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    mask2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    mask3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    mask4 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    mask5 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    mask6 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    mask7 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    mask8 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])

    listMasks = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8]

    x = lim
    y = lim

    for x in range(w - lim):
        for y in range(h - lim):
            if calcMaskRobinson(imgCol, x, y, listMasks) > valLim:
                img.putpixel((x, y), (255, 255, 255))

    return img

def media_pixel(pixel):
    return (pixel[0] + pixel[1] + pixel[2]) / 3

def calcMaskRoberts(img, x, y, mask1, mask2):
    matPxl = matrizPixelsRoberts(img, x, y)

    matGx = multMatrizes(mask1, matPxl)
    matGy = multMatrizes(mask2, matPxl)

    somaGx = somaValMatriz(matGx)
    somaGy = somaValMatriz(matGy)

    quadGx = somaGx * somaGx
    quadGy = somaGy * somaGy

    somaGxGy = quadGx + quadGy

    grad = math.sqrt(somaGxGy)

    return grad

def calcMaskSobel(img, x, y, mask1, mask2):
    tam = mask1.shape[0]
   
    matPxl = matrizPixels(img, x, y, tam)

    matGx = multMatrizes(mask1, matPxl)
    matGy = multMatrizes(mask2, matPxl)

    somaGx = somaValMatriz(matGx)
    somaGy = somaValMatriz(matGy)

    quadGx = somaGx * somaGx
    quadGy = somaGy * somaGy

    somaGxGy = quadGx + quadGy

    grad = math.sqrt(somaGxGy)

    return grad

def calcMaskRobinson(img, x, y, listMask):
    tam = listMask[0].shape[0]
   
    matPxl = matrizPixels(img, x, y, tam)

    listMatMult = [0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(8):
        listMatMult[i] = multMatrizes(listMask[i], matPxl)

    listMatSoma = [0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(8):
        listMatSoma[i] = somaValMatriz(listMatMult[i])

    listMatQuad = [0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(8):
        listMatQuad[i] = listMatSoma[i] * listMatSoma[i] 

    somaTotal = 0

    for i in range(8):
        somaTotal += listMatQuad[i]

    gradTotal = math.sqrt(somaTotal)

    return gradTotal

def matrizPixelsRoberts(img, x, y):
    matResult = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            pxl = img.getpixel((x + i, y + j))
            corResult = (pxl[0] + pxl[1] + pxl[2]) / 3

            matResult[i][j] = int(corResult)

    return matResult

def todosDifZero(img, mask, x, y):
    tam = mask.shape[0]
    lim = int((tam - 1) / 2)

    i2 = -lim
    j2 = -lim
    for i in range(tam):
        for j in range(tam):
            if mask[i][j] == 1:
                pxl = img.getpixel((x + i2, y + j2))
                if int(media_pixel(pxl)) == 0:
                    return False
            j2 += 1
        j2 = -lim
        i2 += 1
    
    return True

def novaImagem(w, h):
    img = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            img.putpixel((x, y), (0, 0, 0))

    return img  

def matrizPixels(imgCol, x, y, tam):
    matResult = np.zeros((tam, tam))

    lim = int((tam - 1) / 2)

    i = -lim 
    j = -lim
    
    i2 = 0
    j2 = 0

    while i <= lim:
        while j <= lim:
            pxl = imgCol.getpixel((x + i, y + j))
            corResult = (pxl[0] + pxl[1] + pxl[2]) / 3

            matResult[i2][j2] = int(corResult)
            j += 1
            j2 += 1

        j = -lim
        j2 = 0

        i += 1
        i2 += 1

    return matResult

def multMatrizes(mat1, mat2):
    tam = mat1.shape[0]

    matResult = np.zeros((tam, tam))

    for x in range(tam):
        for y in range(tam):         
            matResult[x][y] = mat1[x][y] * mat2[x][y]

    return matResult

def somaValMatriz(mat):
    soma = 0
    tam = mat.shape[0]

    for x in range(tam):
        for y in range(tam):
            soma += mat[x][y]

    return soma

if __name__ == "__main__":

    sair = True

    while not sair:
        print('Escolha\n')
        print(' 1 - Cinza\n2 - Cinza ponderado\n3 - Limiarização')
        print(' 4 - Negativo\n5 - Soma\n6 - Subtração')
        print(' 7 - Pontos Salientes\n8 - Dilatação\n9 - Erosão')
        print(' 10 - Abertura\n11 - Fechamento\n12 - Roberts')
        print(' 13 - Sobel\n14 - Robinson\n15 - Sair')

        opcao = input('\nOpção: ')
        limpar()

        if opcao == '1':
            imgOri = Image.open("imgsOriginais\\outono.jpg")
            imgMod = cinza(imgOri)
            imgMod.save("imgsModificadas\\paisagemCinza.jpg")
            show_horizontal(imgOri, imgMod)
        elif opcao == '2':
            imgOri = Image.open("imgsOriginais\\outono.jpg")
            imgMod = cinza_ponderada(imgOri)
            imgMod.save("imgsModificadas\\paisagemCinzaPon.jpg")
            show_horizontal(imgOri, imgMod)
        elif opcao == '3':
            imgOri = Image.open("imgsOriginais\\ferrari1.jpg")
            imgMod = limiarizacao(imgOri)
            imgMod.save("imgsModificadas\\ferrariModLim.jpg")
            show_horizontal(imgOri, imgMod)
        elif opcao == '4':
            imgOri = Image.open("imgsOriginais\\rioDeJaneiro.jpg")
            imgMod = negativa(imgOri)
            imgMod.save("imgsModificadas\\rioNegativo.jpg")
            show_horizontal(imgOri, imgMod)
        elif opcao == '5':
            imgOri1 = Image.open("imgsOriginais\\outono.jpg")
            imgOri2 = Image.open("imgsOriginais\\rioDeJaneiro.jpg")
            imgMod = soma(imgOri1, imgOri2)
            imgMod.save("imgsModificadas\\somaImgs.jpg")
            #imgExt = show_vertical(imgOri1, imgOri2)
            #show_horizontal(imgExt, imgMod)
        elif opcao == '6':
            imgOri1 = Image.open("imgsOriginais\\ferrari1.jpg")
            imgOri2 = Image.open("imgsOriginais\\ferrari2.jpg")
            imgMod = subtracao(imgOri1, imgOri2)
            imgMod.save("imgsModificadas\\ferrarisSub.jpg")
            #show_horizontal(show_vertical(imgOri1, imgOri2), imgMod)
        elif opcao == '7':
            imgOri = Image.open("imgsOriginais\\oncaPintada.jpg")
            imgMod = pontos_salientes(imgOri)
            imgMod.save("imgsModificadas\\oncaSaliente.jpg")
            show_horizontal(imgOri, imgMod)
        elif opcao == '8':
            imgOri = Image.open("imgsOriginais\\prego2.PNG")
            imgMod = dilatacao(limiarizacao(imgOri))
            imgMod.save("imgsModificadas\\prego2Dil.PNG")
            #show_horizontal(imgOri, imgMod)
        elif opcao == '9':
            imgOri = Image.open("imgsOriginais\\prego2.PNG")
            imgMod = erosao(limiarizacao(imgOri))
            imgMod.save("imgsModificadas\\prego2Ero.PNG")
            #show_horizontal(imgOri, imgMod)
        elif opcao == '10':
            imgOri = Image.open("imgsOriginais\\prego2.PNG")
            imgMod = abertura(imgOri)
            imgMod.save("imgsModificadas\\prego2Abert.PNG")
            #show_horizontal(imgOri, imgMod)
        elif opcao == '11':
            imgOri = Image.open("imgsOriginais\\prego2.PNG")
            imgMod = fechamento(imgOri)
            imgMod.save("imgsModificadas\\prego2Fech.PNG")
            #show_horizontal(imgOri, imgMod)
        elif opcao == '12':
            imgOri = Image.open("imgsOriginais\\castelo.jpg")
            imgMod = roberts(imgOri)
            imgMod.save("imgsModificadas\\casteloRober.jpg")
            show_horizontal(imgOri, imgMod)
        elif opcao == '13':
            imgOri = Image.open("imgsOriginais\\castelo.jpg")
            imgMod = sobel(imgOri)
            imgMod.save("imgsModificadas\\casteloSobel.jpg")
            show_horizontal(imgOri, imgMod)
        elif opcao == '14':
            imgOri = Image.open("imgsOriginais\\prego2.PNG")
            imgMod = robinson(imgOri)
            imgMod.save("imgsModificadas\\prego2Robin.PNG")
            show_horizontal(imgOri, imgMod)
        else:
            sair = True

    #end while



    imgOri = Image.open("imgsOriginais\\newYork.jpg")
    imgOri1 = Image.open("imgsOriginais\\oncaPintada.jpg")
    imgMod = roberts(imgOri1)
    imgMod.save("imgsModificadas\\oncaPintadaaaa.jpg")



    # Cinza
    #img = Image.open("imgsOriginais\\outono.jpg")
    #cinza(img).save("imgsModificadas\\paisagemCinza.jpg")

    # Limiarizacao
    #img2 = Image.open("imgsOriginais/castelo.jpg")
    #limiarizacao(img2).save("imgsModificadas/casteloLimiarizada.jpg")

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
    #img9 = Image.open("imgsOriginais\\prego2.PNG")
    #img10 = limiarizacao(img9)
    #dilatacao(img10).save("imgsModificadas\\prego2mod.PNG")

    # Erosao
    #img11 = Image.open("imgsOriginais\\prego2.PNG")
    #img12 = limiarizacao(img11)
    #erosao(img12).save("imgsModificadas\\prego2modEr.PNG")

    # Abertura
    #img13 = Image.open("imgsOriginais\\prego2.PNG")
    #img14 = limiarizacao(img13)
    #abertura(img14).save("imgsModificadas\\prego2modAbert.PNG")

    # Fechamento
    #img15 = Image.open('imgsOriginais/prego2.PNG')
    #img16 = limiarizacao(img15)
    #fechamento(img16).save("imgsModificadas/prego2modFec.PNG")

    # Roberts
    #img17 = Image.open('imgsOriginais/castelo.jpg')
    #img18 = limiarizacao(img17)
    #roberts(img17).save("imgsModificadas/casteloRober.jpg")

    # Sobel
    #img19 = Image.open('imgsOriginais/prego2.PNG')
    #img18 = limiarizacao(img17)
    #sobel(img19).save("imgsModificadas/prego2Sobel.PNG")

    # Robinson
    #img20 = Image.open('imgsOriginais/castelo.jpg')
    #img18 = limiarizacao(img17)
    #robinson(img20).save("imgsModificadas/casteloRobin.jpg")
