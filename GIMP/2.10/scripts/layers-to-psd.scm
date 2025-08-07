(define (layers-to-psd image-paths psd-path)
  (define (add-layers image image-paths)
    (when (not (null? image-paths))
      (let*
        (
          (img-path (car image-paths))
          (layer (car (gimp-file-load-layer RUN-NONINTERACTIVE image img-path)))
          (mrename (car (gimp-item-set-name layer "Background removed")))
        )
        ;; (display img-path) (newline)
        (gimp-image-insert-layer image layer 0 -1)
        (add-layers image (cdr image-paths))
      )
    )
  ) 
  
 
  (let*
    (
      (base-path (car image-paths))
      (image (car (gimp-file-load RUN-NONINTERACTIVE base-path base-path)))
      (drawable (car (gimp-image-get-active-layer image)))   
      (renamed (car (gimp-item-set-name drawable "Original image")))
      (mask (car (gimp-layer-create-mask drawable 2))) ; Add an alpha channel layer mask
    )
    ;; (display base-path) (newline)
    (add-layers image (cdr image-paths))
    ;; LZW compression (1), MSB to LSB fill order (0)
    (define layer (car (gimp-image-get-active-layer image)))
    (define back-layer (car (gimp-image-get-layers image)))
    ;;(define renamed (car (gimp-item-set-name back-layer backgroundremoved)))
    (define masked (car (gimp-layer-create-mask layer 2)))
    (define linked (gimp-item-set-linked back-layer TRUE))
    (define added-mask (gimp-layer-add-mask back-layer masked))
    
    (file-psd-save RUN-NONINTERACTIVE image back-layer psd-path psd-path 1 0)
    (gimp-image-delete image)
  )
)


