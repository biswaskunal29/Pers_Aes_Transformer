import urllib.parse as urlparse
import os
import re

img_url = "http://pbs.twimg.com/profile_images/2994490302/0a5794685603bf7bb4a19133ad87f8da_normal.jpeg"

#path = urlparse.urlparse(img_url).path
#ext = os.path.splitext(path)[1]
#strip = "_normal" + ext
#dp_url_old  = img_url.rstrip(strip)

#re.sub('_x1$', '', x1_field)
       
dp_url_old = re.sub('_normal', '', img_url)
#dp_url = dp_url_old + ext
print(dp_url_old)