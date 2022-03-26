import streamlit as st
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')


def build_sidebar():
    st.sidebar.write("## APS 2021-2")
    nomes = [
        "Thalison Leal Mota",
        "Lucas Pedroso da Cruz Delsin",
        "Fabricio Carlos Torres Junior",
        "Guilherme Duarte Framartino",
        "Jonatan Henrique Ortega",
        ""
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

def put_sticker_on_head(img):
    witch = cv2.imread('witch.png')
    original_witch_h, original_witch_w, witch_channels = witch.shape
    witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)
    ret, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h
        witch_width = int(1.5 * face_w)
        witch_height = int(witch_width * original_witch_h / original_witch_w)
        witch_x1 = face_x2 - int(face_w/2) - int(witch_width/2)
        witch_x2 = witch_x1 + witch_width
        witch_y1 = face_y1 - int(face_h*1.25)
        witch_y2 = witch_y1 + witch_height
        if witch_x1 < 0:
            witch_x1 = 0
        if witch_y1 < 0:
            witch_y1 = 0
        if witch_x2 > img_w:
            witch_x2 = img_w
        if witch_y2 > img_h:
            witch_y2 = img_h
        witch_width = witch_x2 - witch_x1
        witch_height = witch_y2 - witch_y1
        witch = cv2.resize(witch, (witch_width, witch_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (witch_width, witch_height), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (witch_width, witch_height), interpolation=cv2.INTER_AREA)
        roi = img[witch_y1:witch_y2, witch_x1:witch_x2]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
        roi_fg = cv2.bitwise_and(witch, witch, mask=mask_inv)
        dst = cv2.add(roi_bg, roi_fg)
        img[witch_y1:witch_y2, witch_x1:witch_x2] = dst
    return img


def process_types(img, type_selectbox, color_selectbox):
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
        result_image = put_sticker_on_head(img)
    return result_image


def process_image(col_right):
    uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'png'])
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

    with col_right:
        st.markdown('#### Resultado do processamento.')
        img_placeholder = st.empty()

    if uploaded_file:
        img = decode(uploaded_file)

        if type_selectbox == "Indentificar rosto":
            if not detect_faces(img):
                with col_right:
                    st.write('Nenhum rosto encontrado.')

        result_image = process_types(img, type_selectbox, color_selectbox)
        imageRGB = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        set_image(img_placeholder, imageRGB)


def process_video(col_right):
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

    if process:
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            result_image = process_types(img, type_selectbox, color_selectbox)
            cv2.imshow('img', result_image)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
