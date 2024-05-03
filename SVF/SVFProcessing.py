import os
import re
import csv
import json
import glob
import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image



class BSVSpider:
    """
    Climb Baidu Street View from 0°, 90°, 180° and 270°, where the Baidu Street View resolution is set to 512×512, 
    you can modify the URL address in the getImage() function
    """
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)


    # ------------#
    # write csv
    # ------------#
    def writeCsv(self, filepath, data, head=None):
        if head:
            data = [head] + data
        with open(filepath, mode='w', encoding='UTF-8-sig', newline='') as f:
            writer = csv.writer(f)
            for i in data:
                writer.writerow(i)

    # -------------#
    # read csv
    # -------------#
    def readCsv(self, filepath):
        data = []
        if os.path.exists(filepath):
            file_folder = os.path.basename(filepath).split("_")[0]
            with open(filepath, mode='r', encoding='utf-8') as f:
                # ---------------------------------------------------------------#
                # The data read here is returned as a list for each row of data
                # ---------------------------------------------------------------#
                lines = csv.reader(f)
                for line in lines:
                    data.append(line)
            return data, file_folder
        else:
            print('filepath is wrong: {}'.format(filepath))
            return []
    
    def openUrl(_url):
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0.0.0 Safari/537.36"
        }
        response = requests.get(_url, headers=headers)

        # ----------------------------------------------------------#
        # If the status code is 200, 
        # the lifetime server has successfully processed the request 
        # and continues processing the data
        # ----------------------------------------------------------#
        if response.status_code == 200:
            return response.content
        else:
            return None
    
    def getPanoId(self, _lng, _lat):
        # --------------------------------#
        # get svid of baidu streetview
        # --------------------------------#
        url = "https://mapsv0.bdimg.com/?&qt=qsdata&x=%s&y=%s&l=17.031000000000002&action=0&mode=day&t=1530956939770" % (
            str(_lng), str(_lat))
        response = self.openUrl(url).decode("utf8")
    
        if (response == None):
            return None
        reg = r'"id":"(.+?)",'
        pat = re.compile(reg)
        try:
            svid = re.findall(pat, response)[0]
            return svid
        except:
            return None

    def grabBsv(self, _url, _headers=None):
        if _headers == None:
            headers = {
                "sec-ch-ua": '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
                "Referer": "https://map.baidu.com/",
                "sec-ch-ua-mobile": "?0",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/116.0.0.0 Safari/537.36"
            }
        else:
            headers = _headers
        response = requests.get(_url, headers=headers)
        if response.status_code == 200 and response.headers.get("Content-Type") == "image/jpeg":
            return response.content
        else:
            return None
    # ---------------------------------------------------------------------------------------------------------------#
    # Because Baidu Street View is obtained using the Baidu Mercator projection after secondary encryption-->bd09mc
    # ---------------------------------------------------------------------------------------------------------------#
    def wgs2bd09mc(self, wgs_x, wgs_y):
        url = 'https://api.map.baidu.com/geoconv/v1/?coords={}+&from=1&to=6&output=json&ak={}'.format(
            wgs_x + ',' + wgs_y,
            'mYL7zDrHfcb0ziXBqhBOcqFefrbRUnuq'
        )
        res = self.openUrl(url).decode()
        temp = json.loads(res)
        bd09mc_x = 0
        bd09mc_y = 0
        if temp['status'] == 0:
            bd09mc_x = temp['result'][0]['x']
            bd09mc_y = temp['result'][0]['y']
    
        return bd09mc_x, bd09mc_y
    
    def getImage(self, root, input, errorOutput):
        """
        root: the root path of the project
        input: path of the csv file
        errorOutput: the storage file address of the SVF observation point was not successfully crawled
        Note check whether the latitude and longitude in the csv file correspond to wgs_x and wgs_y
        """
        file_name = os.path.basename(os.path.join(root, input)).split("_")[0]
        data = self.readCsv(os.path.join(root, input))
        city_path = os.path.join(root, data[1])
        if not os.path.exists(city_path):
            self.mkdir(city_path)

        filenames_exist = glob.glob1(os.path.join(root, data[1]), "*.jpg")
        # record header
        header = data[0][0]
        # remove header
        new_data = data[0][1:]
        # record the images that failed to be crawled
        error_img = []
        # record the location without svid
        svid_none = []
        # directions, 0 is north
        headings = ['0', '90', '180', '270']
        pitchs = '0'
    
        count = 1

        for i in range(len(new_data)):
            print('Processing No.{} point...'.format(i + 1))
            wgs_x, wgs_y = new_data[i][12], new_data[i][13]
            try:
                bd09mc_x, bd09mc_y = self.wgs2bd09mc(wgs_x, wgs_y)
            except Exception as e:
                print(str(e))
                continue
            flag = True
            for k in range(len(headings)):
                flag = flag and "%s_%s_%s_%s_%s.jpg" % (file_name, wgs_x, wgs_y, headings[k], pitchs) in filenames_exist
            if (flag):
                continue
            svid = self.getPanoId(bd09mc_x, bd09mc_y)
            print("panoid:", svid)

            if svid != None:
                if not os.path.exists(os.path.join(city_path, svid)):
                    self.mkdir(os.path.join(city_path, svid))
                
                for h in range(len(headings)):
                    url = 'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={}&heading={}&pitch=0&width=512&height=512'.format(
                    svid, headings[h]
                    )
                    img = self.grabBsv(url)
                    if img == None:
                        new_data[i].append(headings[h])
                        error_img.append(new_data[i])
    
                    if img != None:
                        with open(os.path.join(city_path, svid) + r'\%s_%s_%s_%s_%s.jpg' % (
                                file_name, wgs_x, wgs_y, headings[h], pitchs),
                                  "wb") as f:
                            f.write(img)
            # sleep 6s
            time.sleep(6)
            count += 1
        # save failed SVF observation points
        if len(error_img) > 0:
            if not os.path.exists(os.path.join(city_path, errorOutput)):
                self.mkdir(os.path.join(city_path, errorOutput))
            self.writeCsv(os.path.join(city_path, errorOutput), error_img, header)


class BSVCalculation:
    def getImage(self, panoid, pos_x, pos_y, zoom):
        """
        panoid: id of the street view image
        pos_x: the position in the x direction
        pos_y: the position in the y direction
        """
        url = "https://mapsv1.bdimg.com/?qt=pdata&sid={}&pos={}_{}&z={}&udt=20200825&from=PC&auth=53CFLyycaQeyKS1gYcbwZ2I%40gAOS283%40uxLTHLzHxRztdw8E62qvyuAMS7IAOUMAvSuDEXcC%40BvkGcuVtvvhguVtvyheuVtvCMGuVtvCQMuxVtE5Wl1GDw8wkv7uvZgMuVtv%40vcuVtvc3CuVtvcPPuVtveGvuzxtwfiKv7uvh3CuVtvhgMuzVVtvrMhuzEtJggagyYxegvcguxLTHLzETzH&seckey=rsKyaODPg5IqfutruCccvLJaKE1ZPiVWCmODAFZhkec%3D%2CiG4_JOnlILoz6AZvcsWhYCcRkcx4Ik8acBLOvhmQT4PqMN33qlRWlUEvM3zq0L01bE-BxYcA1pB1wPwgnI9Fa_Qxmv28v9w0falHeK1T4wkDFLUPoJLhPlFLgOF7h4_gf3vqbeCG34akLjSwdyIZQlmr225GlkPCN5x4RLp0sQm7tL0m-g12M3LfaPO_hngy".format(
                    panoid, pos_x, pos_y, zoom
                )
        try:
            response = requests.get(url)
            if response.status_code == requests.codes.ok:
                file = BytesIO(response.content)
            return file
        except ValueError:
            return None
        
    def getMosaicImage(self, panoid, output512, output256):
        """
        panoid: id of the street view image
        output512: the path to the output image must be specific to the file name and postfix(height: 512)
        output256: the path to the output image must be specific to the file name and postfix(height: 256)
        postfix can be jpeg of png
        """
        tile_size = 512
        num_tilesx = 8
        num_tilesy = 4
        #--------------------------------------------#
        # the total width and height of the mosaic
        #--------------------------------------------#
        mosaic_xsize = tile_size * num_tilesx
        mosaic_ysize = tile_size * num_tilesy
        mosaic_original = Image.new("RGB", (mosaic_xsize, mosaic_ysize), "black")
        black_pixels = 0
        for y in range(0, num_tilesy):
            for x in range(0, num_tilesx):
                imageTile = self.getImage(panoid, y, x, 4)
                if imageTile == None:
                    return ""
                img = Image.open(imageTile)
                # img.save(str(y) + "_" + str(x) + ".png")
                if x == 3:
                    pix_val = list(img.getdata())
                    blk1 = pix_val[tile_size * tile_size - 1]
                    blk2 = pix_val[tile_size * (tile_size - 1)]
                    black_pixels = black_pixels + sum(blk1) + sum(blk2)
                mosaic_original.paste(img, (x * tile_size, y * tile_size, x * tile_size + tile_size, y * tile_size + tile_size))
        x_start = (512 - 128) / 2
        x_size = mosaic_xsize - x_start * 2
        y_size = mosaic_ysize - (512 - 320)
        if black_pixels == 0:
            mosaic_original.crop((x_start, 0, x_start + x_size, y_size))
        mosaic_original = mosaic_original.resize((1024, 512))
        mosaic_crop = mosaic_original.crop((0, 0, 1024, 256))
        mosaic_original.save(output512)
        mosaic_crop.save(output256)

    def panorama2fisheyeDeepLab(input, output, isSegment):
        img = Image.open(input)
        width, height = img.size
        if height == 512:
            img = img.crop((0, 0, width, height / 2))
            width, height = img.size
        red, green, blue = img.split()
        red = np.asarray(red)
        green = np.asarray(green)
        blue = np.asarray(blue)
        #---------------------------------------------------------------------------------------------------#
        # Creates a 3D NumPy array of shape (512, 512, 3) of data type np.uint8 for storing fisheye images
        #---------------------------------------------------------------------------------------------------# 
        fisheye = np.ndarray(shape=(512, 512, 3), dtype=np.uint8)
        fisheye.fill(0)
    
        #--------------------------------# 
        # Normalized to between 0 and 1
        #--------------------------------# 
        x = np.arange(0, 512, dtype=float)
        x = x / 511.0
        x = (x - 0.5) * 2
        x = np.tile(x, (512, 1))
    
        #---------------------------------------------------------------------------------------#
        # Assigns the transpose of x to the variable y to form a 512x512 two-dimensional array
        #---------------------------------------------------------------------------------------#
        y = x.transpose()
        dist2ori = np.sqrt((y * y) + (x * x))
    
        #----------------------------#
        # Convert distance to angle
        #----------------------------#
        angle = dist2ori * 90.0
        angle[np.where(angle <= 0.000000001)] = 0.000000001
        # 加入svf计算的权重
        radian = angle * 3.1415926 / 180.0
        fisheye_weight = np.sin(radian) / (angle / 90.0)
    
        x2 = np.ndarray(shape=(512, 512), dtype=float)
        x2.fill(0.0)
        y2 = np.ndarray(shape=(512, 512), dtype=float)
        y2.fill(1.0)
        cosa = (x * x2 + y * y2) / np.sqrt((x * x + y * y) * (x2 * x2 + y2 * y2))
        lon = np.arccos(cosa) * 180.0 / 3.1415926
        indices = np.where(x > 0)
        lon[indices] = 360.0 - lon[indices]
        lon = 360.0 - lon
        lon = 1.0 - (lon / 360.0)
        outside = np.where(dist2ori > 1)
        lat = dist2ori
        srcx = (lon * (width - 1)).astype(int)
        srcy = (lat * (height - 1)).astype(int)
        srcy[np.where(srcy > 255)] = 0
        indices = (srcx + srcy * width).tolist()
    
        red = np.take(red, np.array(indices))
        green = np.take(green, np.array(indices))
        blue = np.take(blue, np.array(indices))
        red[outside] = 0
        green[outside] = 0
        blue[outside] = 0
    
        svf = -1
        tvf = -1
        bvf = -1
        # RGB(180, 130, 70)
        sky_mask      = 65536 * 180 + 256 * 130 + 70
        # RGB(35, 142, 107)
        tree_mask     = 65536 * 35 + 256 * 142 + 107
        # RGB(70, 70, 70)
        building_mask = 65536 * 70 + 256 * 70 + 70
        if isSegment:
            all_pixels   = 65536 * red + 256 * green + blue
            sky_indices  = np.where(all_pixels == sky_mask)
            tree_indices = np.where(all_pixels == tree_mask)
            building_indices = np.where(all_pixels == building_mask)
    
            background_indices = np.where(all_pixels != 0)
            svf = np.sum(fisheye_weight[sky_indices]) / np.sum(fisheye_weight[background_indices])
            tvf = np.sum(fisheye_weight[tree_indices]) / np.sum(fisheye_weight[background_indices])
            bvf = np.sum(fisheye_weight[building_indices]) / np.sum(fisheye_weight[background_indices])
    
            red[sky_indices]   = 180
            green[sky_indices] = 130
            blue[sky_indices]  = 70

            red[tree_indices]   = 35
            green[tree_indices] = 142
            blue[tree_indices]  = 107

            red[building_indices]   = 70
            green[building_indices] = 70
            blue[building_indices]  = 70
    
        red[outside] = 255
        green[outside] = 255
        blue[outside] = 255
        fisheye = np.dstack((red, green, blue))
        Image.fromarray(fisheye).save(output)
        return [svf, tvf, bvf]
    

    def panorama2fisheyeDeepLabPlus(input, output, isSegment):
        img = Image.open(input)
        width, height = img.size
        if height == 512:
            img = img.crop((0, 0, width, height / 2))
            width, height = img.size
        red, green, blue = img.split()
        red = np.asarray(red)
        green = np.asarray(green)
        blue = np.asarray(blue)
        #---------------------------------------------------------------------------------------------------#
        # Creates a 3D NumPy array of shape (512, 512, 3) of data type np.uint8 for storing fisheye images
        #---------------------------------------------------------------------------------------------------# 
        fisheye = np.ndarray(shape=(512, 512, 3), dtype=np.uint8)
        fisheye.fill(0)
    
        #--------------------------------# 
        # Normalized to between 0 and 1
        #--------------------------------# 
        x = np.arange(0, 512, dtype=float)
        x = x / 511.0
        x = (x - 0.5) * 2
        x = np.tile(x, (512, 1))
    
        #---------------------------------------------------------------------------------------#
        # Assigns the transpose of x to the variable y to form a 512x512 two-dimensional array
        #---------------------------------------------------------------------------------------#
        y = x.transpose()
        dist2ori = np.sqrt((y * y) + (x * x))
    
        #----------------------------#
        # Convert distance to angle
        #----------------------------#
        angle = dist2ori * 90.0
        angle[np.where(angle <= 0.000000001)] = 0.000000001
        # weight
        radian = angle * 3.1415926 / 180.0
        fisheye_weight = np.sin(radian) / (angle / 90.0)
    
        x2 = np.ndarray(shape=(512, 512), dtype=float)
        x2.fill(0.0)
        y2 = np.ndarray(shape=(512, 512), dtype=float)
        y2.fill(1.0)
        cosa = (x * x2 + y * y2) / np.sqrt((x * x + y * y) * (x2 * x2 + y2 * y2))
        lon = np.arccos(cosa) * 180.0 / 3.1415926
        indices = np.where(x > 0)
        lon[indices] = 360.0 - lon[indices]
        lon = 360.0 - lon
        lon = 1.0 - (lon / 360.0)
        outside = np.where(dist2ori > 1)
        lat = dist2ori
        srcx = (lon * (width - 1)).astype(int)
        srcy = (lat * (height - 1)).astype(int)
        srcy[np.where(srcy > 255)] = 0
        indices = (srcx + srcy * width).tolist()
    
        red = np.take(red, np.array(indices))
        green = np.take(green, np.array(indices))
        blue = np.take(blue, np.array(indices))
        red[outside] = 0
        green[outside] = 0
        blue[outside] = 0
    
        svf = -1
        tvf = -1
        bvf = -1
        # RGB(70, 130, 180)
        sky_mask      = 65536 * 70 + 256 * 130 + 180 # 65536 = (512/2)*(512/2)
        # RGB(107, 142, 35)
        tree_mask     = 65536 * 107 + 256 * 142 + 35
        # RGB(70, 70, 70)
        building_mask = 65536 * 70 + 256 * 70 + 70
        if isSegment:
            all_pixels   = 65536 * red + 256 * green + blue
            sky_indices  = np.where(all_pixels == sky_mask)
            tree_indices = np.where(all_pixels == tree_mask)
            building_indices = np.where(all_pixels == building_mask)
    
            background_indices = np.where(all_pixels != 0)
            svf = np.sum(fisheye_weight[sky_indices]) / np.sum(fisheye_weight[background_indices])
            tvf = np.sum(fisheye_weight[tree_indices]) / np.sum(fisheye_weight[background_indices])
            bvf = np.sum(fisheye_weight[building_indices]) / np.sum(fisheye_weight[background_indices])
    
            red[sky_indices]   = 70
            green[sky_indices] = 130
            blue[sky_indices]  = 180

            red[tree_indices]   = 107
            green[tree_indices] = 142
            blue[tree_indices]  = 35

            red[building_indices]   = 70
            green[building_indices] = 70
            blue[building_indices]  = 70
    
        red[outside] = 255
        green[outside] = 255
        blue[outside] = 255
        fisheye = np.dstack((red, green, blue))
        Image.fromarray(fisheye).save(output)
        return [svf, tvf, bvf]
