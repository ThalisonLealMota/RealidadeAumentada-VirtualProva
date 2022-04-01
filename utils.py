import streamlit as st
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')


def build_sidebar():
    st.sidebar.write("## Prova Realidade Virtual e Aumentada")
    nomes = [
        "Thalison Leal Mota",
        "Lucas Pedroso da Cruz Delsin",
        "Fabricio Carlos Torres Junior",
        "Guilherme Duarte Framartino",
        "Jonatan Henrique Ortega",
        "Tiago Faramiglio"
    ]
    st.sidebar.write("## Grupo:")
    for nome in sorted(nomes):
        st.sidebar.write(f"{nome}")

    st.sidebar.write()
    st.sidebar.markdown(
        """ 
        ### Ferramentas utilizadas
        - Python 3.9
        - opencv-python
        - streamlit
        - numpy
        """
    )


def set_image(placeholder, file):
    placeholder.image(file, use_column_width=True)


def decode(image_bytes):
    image_bytes.seek(0)
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def draw_over_the_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
    return img

def overlaw_color_in_image(img, color):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, img.shape[0]), (img.shape[1], 0), color, -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    return img

def put_sticker_on_head(img, stickers):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if stickers[1]:
            hat = cv2.imread('./Imagens/chapeu.png')
            original_hat_h, original_hat_w, hat_channels = hat.shape
            hat_gray = cv2.cvtColor(hat, cv2.COLOR_BGR2GRAY)
            ret, original_mask = cv2.threshold(hat_gray, 10, 255, cv2.THRESH_BINARY_INV)
            original_mask_inv = cv2.bitwise_not(original_mask)
            img_h, img_w = img.shape[:2]
            face_w = w
            face_h = h
            face_x1 = x
            face_x2 = face_x1 + face_w
            face_y1 = y
            face_y2 = face_y1 + face_h
            hat_width = int(1.5 * face_w)
            hat_height = int(hat_width * original_hat_h / original_hat_w)
            hat_x1 = face_x2 - int(face_w/2) - int(hat_width/2)
            hat_x2 = hat_x1 + hat_width
            hat_y1 = face_y1 - int(face_h)
            hat_y2 = hat_y1 + hat_height
            if hat_x1 < 0:
                hat_x1 = 0
            if hat_y1 < 0:
                hat_y1 = 0
            if hat_x2 > img_w:
                hat_x2 = img_w
            if hat_y2 > img_h:
                hat_y2 = img_h
            hat_width = hat_x2 - hat_x1
            hat_height = hat_y2 - hat_y1
            hat = cv2.resize(hat, (hat_width, hat_height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(original_mask, (hat_width, hat_height), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(original_mask_inv, (hat_width, hat_height), interpolation=cv2.INTER_AREA)
            roi = img[hat_y1:hat_y2, hat_x1:hat_x2]
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
            roi_fg = cv2.bitwise_and(hat, hat, mask=mask_inv)
            dst = cv2.add(roi_bg, roi_fg)
            img[hat_y1:hat_y2, hat_x1:hat_x2] = dst
        if stickers[0]:
            sticker = cv2.imread('./Imagens/oculos.png')
            original_sticker_h, original_sticker_w, sticker_channels = sticker.shape
            sticker_gray = cv2.cvtColor(sticker, cv2.COLOR_BGR2GRAY)
            ret, original_mask = cv2.threshold(sticker_gray, 10, 255, cv2.THRESH_BINARY_INV)
            original_mask_inv = cv2.bitwise_not(original_mask)
            img_h, img_w = img.shape[:2]
            face_w = w
            face_h = h
            face_x1 = x
            face_x2 = face_x1 + face_w
            face_y1 = y
            face_y2 = face_y1 + face_h
            sticker_width = int(face_w)
            sticker_height = int(sticker_width * original_sticker_h / original_sticker_w)
            sticker_x1 = face_x2 - int(face_w/2) - int(sticker_width/2)
            sticker_x2 = sticker_x1 + sticker_width
            sticker_y1 = face_y1 + int(face_h/7)
            sticker_y2 = sticker_y1 + sticker_height
            if sticker_x1 < 0:
                sticker_x1 = 0
            if sticker_y1 < 0:
                sticker_y1 = 0
            if sticker_x2 > img_w:
                sticker_x2 = img_w
            if sticker_y2 > img_h:
                sticker_y2 = img_h
            sticker_width = sticker_x2 - sticker_x1
            sticker_height = sticker_y2 - sticker_y1
            sticker = cv2.resize(sticker, (sticker_width, sticker_height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(original_mask, (sticker_width, sticker_height), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(original_mask_inv, (sticker_width, sticker_height), interpolation=cv2.INTER_AREA)
            roi = img[sticker_y1:sticker_y2, sticker_x1:sticker_x2]
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
            roi_fg = cv2.bitwise_and(sticker, sticker, mask=mask_inv)
            dst = cv2.add(roi_bg, roi_fg)
            img[sticker_y1:sticker_y2, sticker_x1:sticker_x2] = dst
        if stickers[2]:
            bigode = cv2.imread('./Imagens/bigode.png')
            original_bigode_h, original_bigode_w, bigode_channels = bigode.shape
            bigode_gray = cv2.cvtColor(bigode, cv2.COLOR_BGR2GRAY)
            ret, original_mask = cv2.threshold(bigode_gray, 10, 255, cv2.THRESH_BINARY_INV)
            original_mask_inv = cv2.bitwise_not(original_mask)
            img_h, img_w = img.shape[:2]
            face_w = w
            face_h = h
            face_x1 = x
            face_x2 = face_x1 + face_w
            face_y1 = y
            face_y2 = face_y1 + face_h
            bigode_width = int(face_w)
            bigode_height = int(bigode_width * original_bigode_h / original_bigode_w)
            bigode_x1 = face_x2 - int(face_w/2) - int(bigode_width/2)
            bigode_x2 = bigode_x1 + bigode_width
            bigode_y1 = face_y1 + int(face_h/3)
            bigode_y2 = bigode_y1 + bigode_height
            if bigode_x1 < 0:
                bigode_x1 = 0
            if bigode_y1 < 0:
                bigode_y1 = 0
            if bigode_x2 > img_w:
                bigode_x2 = img_w
            if bigode_y2 > img_h:
                bigode_y2 = img_h
            bigode_width = bigode_x2 - bigode_x1
            bigode_height = bigode_y2 - bigode_y1
            bigode = cv2.resize(bigode, (bigode_width, bigode_height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(original_mask, (bigode_width, bigode_height), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(original_mask_inv, (bigode_width, bigode_height), interpolation=cv2.INTER_AREA)
            roi = img[bigode_y1:bigode_y2, bigode_x1:bigode_x2]
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
            roi_fg = cv2.bitwise_and(bigode, bigode, mask=mask_inv)
            dst = cv2.add(roi_bg, roi_fg)
            img[bigode_y1:bigode_y2, bigode_x1:bigode_x2] = dst
    
    return img

def process_types(img, type_selectbox, color_selectbox, stickers):
    result_image = img
    if type_selectbox == "Indentificar rosto":
        result_image = draw_over_the_face(img)
    elif type_selectbox == "Alteração de cores":
        if color_selectbox == "Rosa":
            result_image = overlaw_color_in_image(img, (204, 0, 204))
        elif color_selectbox == "Vermelho":
            result_image = overlaw_color_in_image(img, (0, 0, 204))
        elif color_selectbox == "Preto":
            result_image = overlaw_color_in_image(img, (0, 0, 0))
        elif color_selectbox == "Branco":
            result_image = overlaw_color_in_image(img, (200, 200, 200))
        elif color_selectbox == "Cinza":
            result_image = overlaw_color_in_image(img, (60, 60, 60))
    elif type_selectbox == "Colocar adesivo":
        result_image = put_sticker_on_head(img, stickers)
    return result_image


def process_image(col_right):
    oculos = 0
    chapeu = 0
    bigode = 0
    uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'png'])
    type_selectbox = st.selectbox(
        "Escolha o tipo de processamento",
        ("Indentificar rosto", "Alteração de cores", "Colocar adesivo",)
    )
    color_selectbox = None
    if type_selectbox == "Alteração de cores":
        color_selectbox = st.selectbox(
            "Selecione a cor de filtro para a foto",
            ("Rosa", "Vermelho", "Preto", "Branco", "Cinza")
        )

    with col_right:
        st.markdown('#### Resultado do processamento.')
        img_placeholder = st.empty()

    if type_selectbox == "Colocar adesivo":
        with col_right:
            oculos = st.checkbox('Oculos')
            chapeu = st.checkbox('Chapeu')
            bigode = st.checkbox('Bigode')

    if uploaded_file:
        img = decode(uploaded_file)

        if type_selectbox == "Indentificar rosto":
            if not detect_faces(img):
                with col_right:
                    st.write('Nenhum rosto encontrado.')
    
        result_image = process_types(img, type_selectbox, color_selectbox, [oculos, chapeu, bigode])
        imageRGB = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        set_image(img_placeholder, imageRGB)


def process_video(col_right):
    oculos = 0
    chapeu = 0
    bigode = 0
    type_selectbox = st.selectbox(
        "Escolha o tipo de processamento",
        ("Indentificar rosto", "Alteração de cores", "Colocar adesivo")
    )
    color_selectbox = None
    if type_selectbox == "Alteração de cores":
        color_selectbox = st.selectbox(
            "Selecione a cor de filtro para a foto",
            ("Rosa", "Vermelho", "Preto", "Branco", "Cinza")
        )

    process = st.button('Iniciar o vídeo')
    with col_right:
        st.markdown(
            """
            ### Uma janela irá abrir no seu computador com o vídeo rodando.
            """
        )
    

    if type_selectbox == "Colocar adesivo":
        with col_right:
            oculos = st.checkbox('Oculos')
            chapeu = st.checkbox('Chapeu')
            bigode = st.checkbox('Bigode')
            
    if process:
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            result_image = process_types(img, type_selectbox, color_selectbox, [oculos, chapeu, bigode])
            cv2.imshow('img', result_image)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
