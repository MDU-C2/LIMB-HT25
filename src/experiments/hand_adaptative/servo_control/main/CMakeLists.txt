idf_component_register(
    # Archivos fuente a compilar
    SRCS "can_driver.c" 
         "HS422_led.c"
         "main.c"
    
    # Directorios donde buscar headers. El "." indica que busque en la carpeta actual.
    INCLUDE_DIRS "."
    
    # Dependencias de este componente
    # El componente usa twai, ledc y gpio, que están en el driver del ESP-IDF.
    REQUIRES driver log
)