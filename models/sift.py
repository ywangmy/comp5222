import cv2


class SIFT:
    def __init__(self, nfeatures, padding: bool = False):
        self.nfeatures = nfeatures
        self.padding = padding
        self.sift = cv2.xfeatures2d.SIFT_create(
            nfeatures=self.nfeatures, contrastThreshold=0.00001
        )

    def __call__(self):
        sift = self.sift

        # extract keypoints of the image pair using SIFT
        kp1, descs1 = sift.detectAndCompute(image, None)
        # kp2, descs2 = sift.detectAndCompute(warped, None)

        # limit the number of keypoints
        # kp1_num = min(self.nfeatures, len(kp1))

        kp1_num = self.nfeatures
        kp1 = kp1[:kp1_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])

        # confidence of each key point
        scores1_np = np.array([kp.response for kp in kp1])

        if len(kp1_np) < self.nfeatures:
            # print(len(kp1_np))
            if self.padding:
                res = int(self.nfeatures - len(kp1_np))
                pad_kp = (
                    np.random.uniform(size=[res, 2])
                    * (image.shape[0] + image.shape[1])
                    / 2
                )
                pad_scroes1 = np.zeros([res])  # scores := 0
                pad_desc1 = np.zeros((res, 128))

                if len(kp1_np) == 0:
                    kp1_np = pad_kp
                    scores1_np = pad_scroes1
                    descs1 = pad_desc1
                else:
                    kp1_np = np.concatenate([kp1_np, pad_kp], axis=0)
                    scores1_np = np.concatenate([scores1_np, pad_scroes1], axis=0)
                    descs1 = np.concatenate([descs1, pad_desc1], axis=0)
                # print(kp1_np)
                # print(descs1)
                # print(scores1_np)
        kp1_np = kp1_np[:kp1_num, :]
        descs1 = descs1[:kp1_num, :]
        return {"keypoints": kp1_np, "scores": descs1, "descriptors": scores1_np}
