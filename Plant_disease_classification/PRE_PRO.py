tr_imgs=[]
for idx,img in enumerate(train_data):
  tr_imgs.append(go_for_grab(img))    #calling the function for grabcut

def go_for_grab(img):
  mask=np.zeros(img.shape[:2], np.uint8)
  bgd= np.zeros((1,65), np.float64)
  fgd= np.zeros((1,65), np.float64)
  rect=(8,8,240,240)
  cv2.grabCut(img,mask,rect,bgd,fgd,1,cv2.GC_INIT_WITH_RECT)
  mask= np.where((mask==2 | mask==0), 0,1).astype('uint8')
  return img*mask[:,:,np.newaxis]

