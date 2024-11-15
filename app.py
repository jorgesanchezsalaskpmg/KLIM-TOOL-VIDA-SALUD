import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from zipfile import ZipFile
import io
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC



def calcular_proyecciones_ponderadas(df_siniestralidad, df_exposicion_compania):
# Asume que df_exposicion_compania tiene al menos dos filas

    df_exposicion_compania_1 = df_exposicion_compania.copy()

    # Guardar los encabezados actuales en una lista
    headers = df_exposicion_compania_1.columns.tolist()

    # Restablecer los encabezados a los índices por defecto (números enteros)
    df_exposicion_compania_1.columns = range(df_exposicion_compania_1.shape[1])

    # Crear una nueva fila con los encabezados originales
    new_row = pd.DataFrame([headers], columns=df_exposicion_compania_1.columns)

    # Concatenar la nueva fila con el DataFrame original
    # Ignorar el índice para evitar problemas con los índices duplicados
    df_exposicion_compania_1 = pd.concat([new_row, df_exposicion_compania_1], ignore_index=True)


    porcentajes_exposicion = df_exposicion_compania_1.iloc[1]
    porcentajes_exposicion.index = df_siniestralidad.columns[1:20] # Ajuste si es necesario para coincidir con las columnas correctas
    resultados_ponderados = pd.DataFrame()

    for i in range(0, len(df_siniestralidad.columns), 20):
        columnas_escenario = df_siniestralidad.columns[i:i+20]
        df_escenario = df_siniestralidad[columnas_escenario]
        nombre_escenario = columnas_escenario[0]

        for year in df_escenario[nombre_escenario].unique():
            fila = df_escenario[df_escenario[nombre_escenario] == year].iloc[:, 1:]
            siniestralidad_ponderada = (fila.values.flatten() * porcentajes_exposicion).sum()
            resultados_ponderados.loc[year, nombre_escenario] = siniestralidad_ponderada

    resultados_ponderados.columns = [col.replace('Año ', '') for col in resultados_ponderados.columns]
    return resultados_ponderados

# Función para aplicar ajustes y calcular proyecciones para salud OLA CALOR utilizando DataFrames
def calcular_ajustes_y_proyecciones(df_exposicion_compania, df_exposicion_base_dano, df_salud_ola_calor, columna_poblacion):

    df_exposicion_compania_2 = df_exposicion_compania.copy()
    
    # Guardar los encabezados actuales en una lista
    headers = df_exposicion_compania_2.columns.tolist()

    # Restablecer los encabezados a los índices por defecto (números enteros)
    df_exposicion_compania_2.columns = range(df_exposicion_compania_2.shape[1])

    # Crear una nueva fila con los encabezados originales
    new_row = pd.DataFrame([headers], columns=df_exposicion_compania_2.columns)

    # Concatenar la nueva fila con el DataFrame original
    # Ignorar el índice para evitar problemas con los índices duplicados
    df_exposicion_compania_2 = pd.concat([new_row, df_exposicion_compania_2], ignore_index=True)
    
    df_exposicion_compania_traspuesta = df_exposicion_compania_2.transpose()
    df_exposicion_compania_traspuesta.columns = ['CCAA', 'Exposicion']
    #df_exposicion_compania_traspuesta['Exposicion'] = pd.to_numeric(df_exposicion_compania_traspuesta['Exposicion'], errors='coerce')  # Asegurarse de que los datos son numéricos
    df_exposicion_compania_traspuesta.set_index('CCAA', inplace=True)

    df_exposicion_base_dano = df_exposicion_base_dano.set_index('CCAA')

    ajuste_ccaa = (df_exposicion_compania_traspuesta['Exposicion'] / df_exposicion_base_dano[columna_poblacion]) * df_exposicion_base_dano['Daño']
    ajuste_global = ajuste_ccaa.sum()

    df_salud_ola_calor_ajustada = df_salud_ola_calor.copy()
    for escenario in df_salud_ola_calor_ajustada.columns[1:]:  # Asegurarse de que no se incluye la columna con información no numérica
        df_salud_ola_calor_ajustada[escenario] *= ajuste_global

    return df_salud_ola_calor_ajustada

##################################### --- Sección para cargar los archivos base --- ###################################################


# Definición de la función para generar la clave de encriptación
# Definición de la función para generar la clave de encriptación
# Definición de la función para generar la clave de encriptación
def generate_key_from_password(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    generated_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return generated_key

# Función para desencriptar los datos
def decrypt_file(encrypted_data, key):
    fernet = Fernet(key)
    try:
        decrypted_data = fernet.decrypt(encrypted_data)
        return decrypted_data
    except Exception as e:
        st.error(f"No se pudo desencriptar el archivo: {e}")
        return None

# Diccionario con la correspondencia entre la descripción de los archivos y sus nombres
archivos_base = {
    "SALUD DIRECTO +65": "SALUD DIRECTO +65.xlsx",
    "SALUD DIRECTO -65": "SALUD DIRECTO -65.xlsx",
    "SALUD INDIRECTO": "SALUD INDIRECTO.xlsx",
    "SALUD OLA CALOR +65": "SALUD OLA CALOR +65.xlsx",
    "SALUD OLA CALOR -65": "SALUD OLA CALOR -65.xlsx",
    "SALUD OLA CALOR Exposición Base Daño +65": "SALUD OLA CALOR Exposición Base Daño +65.xlsx",
    "SALUD OLA CALOR Exposición Base Daño -65": "SALUD OLA CALOR Exposición Base Daño -65.xlsx",
    "VIDA DIRECTO +65": "VIDA DIRECTO +65.xlsx",
    "VIDA DIRECTO -65": "VIDA DIRECTO -65.xlsx",
    "VIDA INDIRECTO": "VIDA INDIRECTO.xlsx",
    "VIDA OLA CALOR +65": "VIDA OLA CALOR +65.xlsx",
    "VIDA OLA CALOR -65": "VIDA OLA CALOR -65.xlsx",
    "VIDA OLA CALOR Exposición Base Daño +65": "VIDA OLA CALOR Exposición Base Daño +65.xlsx",
    "VIDA OLA CALOR Exposición Base Daño -65": "VIDA OLA CALOR Exposición Base Daño -65.xlsx",
}

# Inicializar el diccionario en el estado de la sesión si aún no existe
if 'archivos_base' not in st.session_state:
    st.session_state['archivos_base'] = {}

st.header("Cargar Archivos Base")

# Input de contraseña
password_input = st.text_input("Introduce la contraseña para desencriptar los archivos:", type="password")

if st.button("Desencriptar y cargar archivos base") and password_input:
    # Obtener la 'salt' de los secretos de Streamlit
    salt = base64.b64decode(st.secrets["MY_SALT_SECRET_KEY"])
    key = generate_key_from_password(password_input, salt)

    # Localizar el archivo encriptado en la misma carpeta del script
    encrypted_zip_path = 'inputs_base.zip.enc'
    
    # Leer y desencriptar el contenido del archivo .zip.enc
    with open(encrypted_zip_path, 'rb') as file:
        encrypted_data = file.read()
    decrypted_zip = decrypt_file(encrypted_data, key)
    
    if decrypted_zip:
        with ZipFile(io.BytesIO(decrypted_zip), 'r') as zip_ref:
            for descripcion, nombre_archivo in archivos_base.items():
                archivo_zip_ruta = f"Inputs Base/{nombre_archivo}"
                if archivo_zip_ruta in zip_ref.namelist():
                    with zip_ref.open(archivo_zip_ruta) as file:
                        content = io.BytesIO(file.read())
                        df = pd.read_excel(content)
                        st.session_state['archivos_base'][descripcion] = df
                        st.success(f"{descripcion} cargado correctamente.")
                else:
                    st.error(f"No se pudo encontrar el archivo {nombre_archivo}")

    else:
        st.error("No se pudo desencriptar los archivos con esa contraseña.")

# Crear un expander que inicialmente está cerrado
with st.expander("Haz clic para expandir y ver la aplicación de salud completa", expanded=False):

    st.title("KLIM TOOL SALUD")

    st.image('logo.PNG', width=120)



    ######################################## --- Sección para cargar los archivos de la compañía --- ############################3
    st.header('Carga de archivos de salud de la compañía')

    salud_exposicion_plus65 = st.file_uploader("Importar SALUD Exposición +65 (Excel)", type=['xlsx'], key='salud_exposicion_plus65')
    salud_exposicion_minus65 = st.file_uploader("Importar SALUD Exposición -65 (Excel)", type=['xlsx'], key='salud_exposicion_minus65')

    if salud_exposicion_plus65 and salud_exposicion_minus65:
        try:
            with st.spinner("Procesando archivos de la compañía..."):
                df_salud_exposicion_plus65 = pd.read_excel(salud_exposicion_plus65)
                df_salud_exposicion_minus65 = pd.read_excel(salud_exposicion_minus65)
                st.session_state['df_salud_exposicion_plus65'] = df_salud_exposicion_plus65
                st.session_state['df_salud_exposicion_minus65'] = df_salud_exposicion_minus65
                st.success('Los archivos de la compañía se cargaron y procesaron correctamente.')
        except Exception as e:
            st.error(f"Error al cargar archivos de la compañía: {e}")
    else:
        if salud_exposicion_plus65 is None or salud_exposicion_minus65 is None:
            st.warning('Es necesario importar ambos archivos de la compañía para proceder con los cálculos.')

    #################################### VISUALIZAR ###############################################3

    st.header('Visualizar DataFrames Salud')

    # Claves para archivos base y archivos de la compañía
    claves_archivos_base = list(archivos_base.keys())  
    claves_df_exposicion = ['df_salud_exposicion_plus65', 'df_salud_exposicion_minus65']

    # Crear lista de DataFrames disponibles para visualizar a partir de los cargados anteriormente
    dataframes_disponibles = claves_archivos_base

    # Agregar los DataFrames de la compañía si se encuentran en la sesión
    if 'df_salud_exposicion_plus65' in st.session_state:
        dataframes_disponibles.append('SALUD Exposición +65')
    if 'df_salud_exposicion_minus65' in st.session_state:
        dataframes_disponibles.append('SALUD Exposición -65')

    selected_df_name = st.selectbox('Selecciona un DataFrame de Salud para visualizar:', dataframes_disponibles)

    # Acción del botón para mostrar el DataFrame seleccionado
    if st.button('Mostrar DataFrame Salud'):
        if selected_df_name in st.session_state['archivos_base']:
            st.write(f"DataFrame {selected_df_name}:")
            st.dataframe(st.session_state['archivos_base'][selected_df_name])
        elif selected_df_name == 'SALUD Exposición +65':
            st.write(f"DataFrame de exposición compañía +65:")
            st.dataframe(st.session_state['df_salud_exposicion_plus65'])
        elif selected_df_name == 'SALUD Exposición -65':
            st.write(f"DataFrame de exposición compañía -65:")
            st.dataframe(st.session_state['df_salud_exposicion_minus65'])
        else:
            st.error(f"El DataFrame {selected_df_name} no está cargado en la sesión.")

    ###################################### Aquí continúa el código con la lógica para realizar los cálculos y proyecciones...#########################################

    def create_excel_bytesio(df, filename, zip_obj):
        # Creamos un objeto BytesIO que representará el archivo Excel
        excel_bytesio = io.BytesIO()
        # Write dataframe to the BytesIO object
        with pd.ExcelWriter(excel_bytesio, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)  # El guardado se realiza al cerrar el bloque 'with'
        excel_bytesio.seek(0)  # Mueve el cursor al inicio del stream
        # Write BytesIO object to the zipfile
        zip_obj.writestr(filename, excel_bytesio.getvalue())

    st.header('Ejecutar Proyecciones de Salud y Exportar a Excel')

    def ejecutar_calculos():
        try:
            # Verificar que los archivos base y de compañía se han cargado correctamente
            if 'archivos_base' not in st.session_state or \
            'df_salud_exposicion_plus65' not in st.session_state or \
            'df_salud_exposicion_minus65' not in st.session_state:
                raise ValueError("Debes cargar todos los archivos necesarios antes de ejecutar los cálculos.")
            
            # Proyecciones ponderadas para SALUD DIRECTO +65 y -65
            resultados_directo_mas65 = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["SALUD DIRECTO +65"],
                st.session_state['df_salud_exposicion_plus65']
            )
            
            resultados_directo_menos65 = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["SALUD DIRECTO -65"],
                st.session_state['df_salud_exposicion_minus65']
            )
            
            # Proyecciones ponderadas para SALUD INDIRECTO
            resultados_indirecto_mas65 = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["SALUD INDIRECTO"],
                st.session_state['df_salud_exposicion_plus65']
            )
            
            resultados_indirecto_menos65 = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["SALUD INDIRECTO"],
                st.session_state['df_salud_exposicion_minus65']
            )
            
            # Ajustes y proyecciones para SALUD OLA CALOR +65 y -65
            resultados_salud_ola_calor_plus65 = calcular_ajustes_y_proyecciones(
                st.session_state['df_salud_exposicion_plus65'].copy(),
                st.session_state['archivos_base']["SALUD OLA CALOR Exposición Base Daño +65"],
                st.session_state['archivos_base']["SALUD OLA CALOR +65"],
                'Poblacion España +65'
            )
            
            resultados_salud_ola_calor_menos65 = calcular_ajustes_y_proyecciones(
                st.session_state['df_salud_exposicion_minus65'].copy(),
                st.session_state['archivos_base']["SALUD OLA CALOR Exposición Base Daño -65"],
                st.session_state['archivos_base']["SALUD OLA CALOR -65"],
                'Poblacion España -65'
            )

            # Creamos un objeto BytesIO que representará el archivo zip
            zip_bytesio = io.BytesIO()
            # Creamos un objeto ZipFile con nuestro BytesIO como archivo
            with ZipFile(zip_bytesio, 'a') as zip_file:
                # Agregamos cada archivo Excel al archivo zip
                create_excel_bytesio(resultados_directo_mas65, 'Resultados_Directo_Mas65.xlsx', zip_file)
                create_excel_bytesio(resultados_directo_menos65, 'Resultados_Directo_Menos65.xlsx', zip_file)
                create_excel_bytesio(resultados_indirecto_mas65, 'Resultados_Indirecto_Mas65.xlsx', zip_file)
                create_excel_bytesio(resultados_indirecto_menos65, 'Resultados_Indirecto_Menos65.xlsx', zip_file)
                create_excel_bytesio(resultados_salud_ola_calor_plus65, 'Resultados_Ola_Calor_Plus65.xlsx', zip_file)
                create_excel_bytesio(resultados_salud_ola_calor_menos65, 'Resultados_Ola_Calor_Menos65.xlsx', zip_file)
                
            zip_bytesio.seek(0)
            
            st.success("Proyecciones calculadas con éxito.")

            # Creamos un botón de descarga para el archivo zip
            st.download_button(
                label="Descargar todos los resultados como ZIP",
                data=zip_bytesio,
                file_name="Resultados.zip",
                mime='application/zip'
            )
            
        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"Se ha producido un error durante el cálculo de las proyecciones: {e}")

    if st.button('Calcular Proyecciones Salud'):
        ejecutar_calculos()


    st.header('Guardar Resultados de Salud en Streamlit')

    def guardar_resultados_streamlit():
        try:
            # Verificar que los archivos base y de compañía se han cargado correctamente
            if 'archivos_base' not in st.session_state or 'df_salud_exposicion_plus65' not in st.session_state or 'df_salud_exposicion_minus65' not in st.session_state:
                raise ValueError("Debes cargar todos los archivos necesarios antes de ejecutar los cálculos.")

            # Inicializa el diccionario de resultados en st.session_state si todavía no existe
            if 'resultados' not in st.session_state:
                st.session_state['resultados'] = {}

            # Proyecciones ponderadas para SALUD DIRECTO +65 y -65
            # ...
            # Simplemente añadir al final de cada bloque de cálculo:

            # Para salud DIRECTO +65
            st.session_state['resultados']['Resultados_Directo_Mas65'] = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["SALUD DIRECTO +65"],
                st.session_state['df_salud_exposicion_plus65']
            )

            # Para SALUD DIRECTO -65
            st.session_state['resultados']['Resultados_Directo_Menos65'] = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["SALUD DIRECTO -65"],
                st.session_state['df_salud_exposicion_minus65']
            )

            # ... Continúa con el resto de los resultados
            # Para SALUD INDIRECTO +65
            st.session_state['resultados']['Resultados_Indirecto_Mas65'] = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["SALUD INDIRECTO"],
                st.session_state['df_salud_exposicion_plus65']
            )

            # Para SALUD INDIRECTO -65
            st.session_state['resultados']['Resultados_Indirecto_Menos65'] = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["SALUD INDIRECTO"],
                st.session_state['df_salud_exposicion_minus65']
            )

            # Ajustes y proyecciones para SALUD OLA CALOR +65 y -65
            # Para SALUD OLA CALOR +65
            st.session_state['resultados']['Resultados_Ola_Calor_Plus65'] = calcular_ajustes_y_proyecciones(
                st.session_state['df_salud_exposicion_plus65'].copy(),
                st.session_state['archivos_base']["SALUD OLA CALOR Exposición Base Daño +65"],
                st.session_state['archivos_base']["SALUD OLA CALOR +65"],
                'Poblacion España +65'
            )
            
            # Para SALUD OLA CALOR -65
            st.session_state['resultados']['Resultados_Ola_Calor_Menos65'] = calcular_ajustes_y_proyecciones(
                st.session_state['df_salud_exposicion_minus65'].copy(),
                st.session_state['archivos_base']["SALUD OLA CALOR Exposición Base Daño -65"],
                st.session_state['archivos_base']["SALUD OLA CALOR -65"],
                'Poblacion España -65'
            )

            st.success("Proyecciones calculadas y guardadas con éxito en la sesión.")

        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"Se ha producido un error durante el cálculo de las proyecciones: {e}")

    if st.button('Guardar Resultados de Salud en Streamlit'):
        guardar_resultados_streamlit()

    ########### Visualizar Output ##################3

    import matplotlib.pyplot as plt

    st.header('Visualización de Resultados de Salud en Gráfico')

    opciones_resultados = [
        'Resultados_Directo_Mas65',
        'Resultados_Directo_Menos65',
        'Resultados_Indirecto_Mas65',
        'Resultados_Indirecto_Menos65',
        'Resultados_Ola_Calor_Plus65',
        'Resultados_Ola_Calor_Menos65'
    ]

    # Una variedad de colores en tonos azules y algunos morados/rosas.
    colores = [
        '#1f77b4',  # Azul moderadamente oscuro
        '#aec7e8',  # Azul claro
        '#3498db',  # Azul claro (más cerca al cian)
        '#5dade2',  # Azul celeste

        # Morados/rosas 
        '#e74c3c',  # Rosa saludable
        '#f1948a',  # Rosa pálido
        '#c39bd3',  # Morado pálido
    ]

    opcion_seleccionada = st.selectbox('Seleccione un resultado de salud para visualizar en gráfico:', opciones_resultados)

    # Almacenaremos la figura creada para poder cerrarla después de mostrarla. Esto previene el error 'RuntimeError: main thread is not in main loop' de Tkinter.
    fig = plt.figure(figsize=(10, 5))

    if st.button('Mostrar Gráfico'):
        if opcion_seleccionada in st.session_state['resultados']:
            df_seleccionado = st.session_state['resultados'][opcion_seleccionada]

            # Si 'Año' no es una columna, suponemos que está en el índice y reseteamos el índice.
            if 'Año' not in df_seleccionado.columns:
                df_seleccionado.reset_index(inplace=True)
                df_seleccionado.rename(columns={'index': 'Año'}, inplace=True)

            # Creamos la figura fuera del botón para evitar posibles errores en subprocesos.
            for i, columna in enumerate(df_seleccionado.columns[1:]):  # Saltamos la columna 'Año'
                plt.plot(df_seleccionado['Año'], df_seleccionado[columna], label=columna, color=colores[i % len(colores)])

            plt.title('Proyecciones por Escenario')
            plt.xlabel('Año')
            plt.ylabel('Valor Proyectado')
            plt.legend()
            #plt.grid(True)
            
            st.pyplot(fig)
            plt.clf()

################################ VIDA ####################################

# Crear un expander que inicialmente está cerrado
with st.expander("Haz clic para expandir y ver la aplicación de vida completa", expanded=False):
    import streamlit as st
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from zipfile import ZipFile
    import io
    from cryptography.fernet import Fernet
    import base64
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC



    def calcular_proyecciones_ponderadas(df_siniestralidad, df_exposicion_compania):
    # Asume que df_exposicion_compania tiene al menos dos filas

        df_exposicion_compania_1 = df_exposicion_compania.copy()

        # Guardar los encabezados actuales en una lista
        headers = df_exposicion_compania_1.columns.tolist()

        # Restablecer los encabezados a los índices por defecto (números enteros)
        df_exposicion_compania_1.columns = range(df_exposicion_compania_1.shape[1])

        # Crear una nueva fila con los encabezados originales
        new_row = pd.DataFrame([headers], columns=df_exposicion_compania_1.columns)

        # Concatenar la nueva fila con el DataFrame original
        # Ignorar el índice para evitar problemas con los índices duplicados
        df_exposicion_compania_1 = pd.concat([new_row, df_exposicion_compania_1], ignore_index=True)


        porcentajes_exposicion = df_exposicion_compania_1.iloc[1]
        porcentajes_exposicion.index = df_siniestralidad.columns[1:20] # Ajuste si es necesario para coincidir con las columnas correctas
        resultados_ponderados = pd.DataFrame()

        for i in range(0, len(df_siniestralidad.columns), 20):
            columnas_escenario = df_siniestralidad.columns[i:i+20]
            df_escenario = df_siniestralidad[columnas_escenario]
            nombre_escenario = columnas_escenario[0]

            for year in df_escenario[nombre_escenario].unique():
                fila = df_escenario[df_escenario[nombre_escenario] == year].iloc[:, 1:]
                siniestralidad_ponderada = (fila.values.flatten() * porcentajes_exposicion).sum()
                resultados_ponderados.loc[year, nombre_escenario] = siniestralidad_ponderada

        resultados_ponderados.columns = [col.replace('Año ', '') for col in resultados_ponderados.columns]
        return resultados_ponderados

    # Función para aplicar ajustes y calcular proyecciones para VIDA OLA CALOR utilizando DataFrames
    def calcular_ajustes_y_proyecciones(df_exposicion_compania, df_exposicion_base_dano, df_vida_ola_calor, columna_poblacion):

        df_exposicion_compania_2 = df_exposicion_compania.copy()
        
        # Guardar los encabezados actuales en una lista
        headers = df_exposicion_compania_2.columns.tolist()

        # Restablecer los encabezados a los índices por defecto (números enteros)
        df_exposicion_compania_2.columns = range(df_exposicion_compania_2.shape[1])

        # Crear una nueva fila con los encabezados originales
        new_row = pd.DataFrame([headers], columns=df_exposicion_compania_2.columns)

        # Concatenar la nueva fila con el DataFrame original
        # Ignorar el índice para evitar problemas con los índices duplicados
        df_exposicion_compania_2 = pd.concat([new_row, df_exposicion_compania_2], ignore_index=True)
        
        df_exposicion_compania_traspuesta = df_exposicion_compania_2.transpose()
        df_exposicion_compania_traspuesta.columns = ['CCAA', 'Exposicion']
        #df_exposicion_compania_traspuesta['Exposicion'] = pd.to_numeric(df_exposicion_compania_traspuesta['Exposicion'], errors='coerce')  # Asegurarse de que los datos son numéricos
        df_exposicion_compania_traspuesta.set_index('CCAA', inplace=True)

        df_exposicion_base_dano = df_exposicion_base_dano.set_index('CCAA')

        ajuste_ccaa = (df_exposicion_compania_traspuesta['Exposicion'] / df_exposicion_base_dano[columna_poblacion]) * df_exposicion_base_dano['Daño']
        ajuste_global = ajuste_ccaa.sum()

        df_vida_ola_calor_ajustada = df_vida_ola_calor.copy()
        for escenario in df_vida_ola_calor_ajustada.columns[1:]:  # Asegurarse de que no se incluye la columna con información no numérica
            df_vida_ola_calor_ajustada[escenario] *= ajuste_global

        return df_vida_ola_calor_ajustada

    st.title("KLIM TOOL VIDA")

    st.image('logo.PNG', width=120)


    ######################################## --- Sección para cargar los archivos de la compañía --- ############################3
    st.header('Carga de archivos de la compañía')

    vida_exposicion_plus65 = st.file_uploader("Importar VIDA Exposición +65 (Excel)", type=['xlsx'], key='vida_exposicion_plus65')
    vida_exposicion_minus65 = st.file_uploader("Importar VIDA Exposición -65 (Excel)", type=['xlsx'], key='vida_exposicion_minus65')

    if vida_exposicion_plus65 and vida_exposicion_minus65:
        try:
            with st.spinner("Procesando archivos de la compañía..."):
                df_vida_exposicion_plus65 = pd.read_excel(vida_exposicion_plus65)
                df_vida_exposicion_minus65 = pd.read_excel(vida_exposicion_minus65)
                st.session_state['df_vida_exposicion_plus65'] = df_vida_exposicion_plus65
                st.session_state['df_vida_exposicion_minus65'] = df_vida_exposicion_minus65
                st.success('Los archivos de la compañía se cargaron y procesaron correctamente.')
        except Exception as e:
            st.error(f"Error al cargar archivos de la compañía: {e}")
    else:
        if vida_exposicion_plus65 is None or vida_exposicion_minus65 is None:
            st.warning('Es necesario importar ambos archivos de la compañía para proceder con los cálculos.')

    #################################### VISUALIZAR ###############################################3

    st.header('Visualizar DataFrames')

    # Claves para archivos base y archivos de la compañía
    claves_archivos_base = list(archivos_base.keys())  
    claves_df_exposicion = ['df_vida_exposicion_plus65', 'df_vida_exposicion_minus65']

    # Crear lista de DataFrames disponibles para visualizar a partir de los cargados anteriormente
    dataframes_disponibles = claves_archivos_base

    # Agregar los DataFrames de la compañía si se encuentran en la sesión
    if 'df_vida_exposicion_plus65' in st.session_state:
        dataframes_disponibles.append('VIDA Exposición +65')
    if 'df_vida_exposicion_minus65' in st.session_state:
        dataframes_disponibles.append('VIDA Exposición -65')

    selected_df_name = st.selectbox('Selecciona un DataFrame para visualizar:', dataframes_disponibles)

    # Acción del botón para mostrar el DataFrame seleccionado
    if st.button('Mostrar DataFrame'):
        if selected_df_name in st.session_state['archivos_base']:
            st.write(f"DataFrame {selected_df_name}:")
            st.dataframe(st.session_state['archivos_base'][selected_df_name])
        elif selected_df_name == 'VIDA Exposición +65':
            st.write(f"DataFrame de exposición compañía +65:")
            st.dataframe(st.session_state['df_vida_exposicion_plus65'])
        elif selected_df_name == 'VIDA Exposición -65':
            st.write(f"DataFrame de exposición compañía -65:")
            st.dataframe(st.session_state['df_vida_exposicion_minus65'])
        else:
            st.error(f"El DataFrame {selected_df_name} no está cargado en la sesión.")

    ###################################### Aquí continúa el código con la lógica para realizar los cálculos y proyecciones...#########################################

    def create_excel_bytesio(df, filename, zip_obj):
        # Creamos un objeto BytesIO que representará el archivo Excel
        excel_bytesio = io.BytesIO()
        # Write dataframe to the BytesIO object
        with pd.ExcelWriter(excel_bytesio, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)  # El guardado se realiza al cerrar el bloque 'with'
        excel_bytesio.seek(0)  # Mueve el cursor al inicio del stream
        # Write BytesIO object to the zipfile
        zip_obj.writestr(filename, excel_bytesio.getvalue())

    st.header('Ejecutar Proyecciones y Exportar a Excel')

    def ejecutar_calculos():
        try:
            # Verificar que los archivos base y de compañía se han cargado correctamente
            if 'archivos_base' not in st.session_state or \
            'df_vida_exposicion_plus65' not in st.session_state or \
            'df_vida_exposicion_minus65' not in st.session_state:
                raise ValueError("Debes cargar todos los archivos necesarios antes de ejecutar los cálculos.")
            
            # Proyecciones ponderadas para VIDA DIRECTO +65 y -65
            resultados_directo_mas65 = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["VIDA DIRECTO +65"],
                st.session_state['df_vida_exposicion_plus65']
            )
            
            resultados_directo_menos65 = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["VIDA DIRECTO -65"],
                st.session_state['df_vida_exposicion_minus65']
            )
            
            # Proyecciones ponderadas para VIDA INDIRECTO
            resultados_indirecto_mas65 = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["VIDA INDIRECTO"],
                st.session_state['df_vida_exposicion_plus65']
            )
            
            resultados_indirecto_menos65 = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["VIDA INDIRECTO"],
                st.session_state['df_vida_exposicion_minus65']
            )
            
            # Ajustes y proyecciones para VIDA OLA CALOR +65 y -65
            resultados_vida_ola_calor_plus65 = calcular_ajustes_y_proyecciones(
                st.session_state['df_vida_exposicion_plus65'].copy(),
                st.session_state['archivos_base']["VIDA OLA CALOR Exposición Base Daño +65"],
                st.session_state['archivos_base']["VIDA OLA CALOR +65"],
                'Poblacion España +65'
            )
            
            resultados_vida_ola_calor_menos65 = calcular_ajustes_y_proyecciones(
                st.session_state['df_vida_exposicion_minus65'].copy(),
                st.session_state['archivos_base']["VIDA OLA CALOR Exposición Base Daño -65"],
                st.session_state['archivos_base']["VIDA OLA CALOR -65"],
                'Poblacion España -65'
            )

            # Creamos un objeto BytesIO que representará el archivo zip
            zip_bytesio = io.BytesIO()
            # Creamos un objeto ZipFile con nuestro BytesIO como archivo
            with ZipFile(zip_bytesio, 'a') as zip_file:
                # Agregamos cada archivo Excel al archivo zip
                create_excel_bytesio(resultados_directo_mas65, 'Resultados_Directo_Mas65.xlsx', zip_file)
                create_excel_bytesio(resultados_directo_menos65, 'Resultados_Directo_Menos65.xlsx', zip_file)
                create_excel_bytesio(resultados_indirecto_mas65, 'Resultados_Indirecto_Mas65.xlsx', zip_file)
                create_excel_bytesio(resultados_indirecto_menos65, 'Resultados_Indirecto_Menos65.xlsx', zip_file)
                create_excel_bytesio(resultados_vida_ola_calor_plus65, 'Resultados_Ola_Calor_Plus65.xlsx', zip_file)
                create_excel_bytesio(resultados_vida_ola_calor_menos65, 'Resultados_Ola_Calor_Menos65.xlsx', zip_file)
                
            zip_bytesio.seek(0)
            
            st.success("Proyecciones calculadas con éxito.")

            # Creamos un botón de descarga para el archivo zip
            st.download_button(
                label="Descargar todos los resultados como ZIP",
                data=zip_bytesio,
                file_name="Resultados.zip",
                mime='application/zip'
            )
            
        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"Se ha producido un error durante el cálculo de las proyecciones: {e}")

    if st.button('Calcular Proyecciones'):
        ejecutar_calculos()


    st.header('Guardar Resultados en Streamlit')

    def guardar_resultados_streamlit():
        try:
            # Verificar que los archivos base y de compañía se han cargado correctamente
            if 'archivos_base' not in st.session_state or 'df_vida_exposicion_plus65' not in st.session_state or 'df_vida_exposicion_minus65' not in st.session_state:
                raise ValueError("Debes cargar todos los archivos necesarios antes de ejecutar los cálculos.")

            # Inicializa el diccionario de resultados en st.session_state si todavía no existe
            if 'resultados' not in st.session_state:
                st.session_state['resultados'] = {}

            # Proyecciones ponderadas para VIDA DIRECTO +65 y -65
            # ...
            # Simplemente añadir al final de cada bloque de cálculo:

            # Para VIDA DIRECTO +65
            st.session_state['resultados']['Resultados_Directo_Mas65'] = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["VIDA DIRECTO +65"],
                st.session_state['df_vida_exposicion_plus65']
            )

            # Para VIDA DIRECTO -65
            st.session_state['resultados']['Resultados_Directo_Menos65'] = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["VIDA DIRECTO -65"],
                st.session_state['df_vida_exposicion_minus65']
            )

            # ... Continúa con el resto de los resultados
            # Para VIDA INDIRECTO +65
            st.session_state['resultados']['Resultados_Indirecto_Mas65'] = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["VIDA INDIRECTO"],
                st.session_state['df_vida_exposicion_plus65']
            )

            # Para VIDA INDIRECTO -65
            st.session_state['resultados']['Resultados_Indirecto_Menos65'] = calcular_proyecciones_ponderadas(
                st.session_state['archivos_base']["VIDA INDIRECTO"],
                st.session_state['df_vida_exposicion_minus65']
            )

            # Ajustes y proyecciones para VIDA OLA CALOR +65 y -65
            # Para VIDA OLA CALOR +65
            st.session_state['resultados']['Resultados_Ola_Calor_Plus65'] = calcular_ajustes_y_proyecciones(
                st.session_state['df_vida_exposicion_plus65'].copy(),
                st.session_state['archivos_base']["VIDA OLA CALOR Exposición Base Daño +65"],
                st.session_state['archivos_base']["VIDA OLA CALOR +65"],
                'Poblacion España +65'
            )
            
            # Para VIDA OLA CALOR -65
            st.session_state['resultados']['Resultados_Ola_Calor_Menos65'] = calcular_ajustes_y_proyecciones(
                st.session_state['df_vida_exposicion_minus65'].copy(),
                st.session_state['archivos_base']["VIDA OLA CALOR Exposición Base Daño -65"],
                st.session_state['archivos_base']["VIDA OLA CALOR -65"],
                'Poblacion España -65'
            )

            st.success("Proyecciones calculadas y guardadas con éxito en la sesión.")

        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"Se ha producido un error durante el cálculo de las proyecciones: {e}")

    if st.button('Guardar Resultados en Streamlit'):
        guardar_resultados_streamlit()

    ########### Visualizar Output ##################3

    import matplotlib.pyplot as plt

    st.header('Visualización de Resultados en Gráfico')

    opciones_resultados = [
        'Resultados_Directo_Mas65',
        'Resultados_Directo_Menos65',
        'Resultados_Indirecto_Mas65',
        'Resultados_Indirecto_Menos65',
        'Resultados_Ola_Calor_Plus65',
        'Resultados_Ola_Calor_Menos65'
    ]

    # Una variedad de colores en tonos azules y algunos morados/rosas.
    colores = [
        '#1f77b4',  # Azul moderadamente oscuro
        '#aec7e8',  # Azul claro
        '#3498db',  # Azul claro (más cerca al cian)
        '#5dade2',  # Azul celeste

        # Morados/rosas 
        '#e74c3c',  # Rosa saludable
        '#f1948a',  # Rosa pálido
        '#c39bd3',  # Morado pálido
    ]

    opcion_seleccionada = st.selectbox('Seleccione un resultado para visualizar en gráfico:', opciones_resultados)

    # Almacenaremos la figura creada para poder cerrarla después de mostrarla. Esto previene el error 'RuntimeError: main thread is not in main loop' de Tkinter.
    fig = plt.figure(figsize=(10, 5))

    if st.button('Mostrar Gráfico'):
        if opcion_seleccionada in st.session_state['resultados']:
            df_seleccionado = st.session_state['resultados'][opcion_seleccionada]

            # Si 'Año' no es una columna, suponemos que está en el índice y reseteamos el índice.
            if 'Año' not in df_seleccionado.columns:
                df_seleccionado.reset_index(inplace=True)
                df_seleccionado.rename(columns={'index': 'Año'}, inplace=True)

            # Creamos la figura fuera del botón para evitar posibles errores en subprocesos.
            for i, columna in enumerate(df_seleccionado.columns[1:]):  # Saltamos la columna 'Año'
                plt.plot(df_seleccionado['Año'], df_seleccionado[columna], label=columna, color=colores[i % len(colores)])

            plt.title('Proyecciones por Escenario')
            plt.xlabel('Año')
            plt.ylabel('Valor Proyectado')
            plt.legend()
            #plt.grid(True)
            
            st.pyplot(fig)
            plt.clf()
