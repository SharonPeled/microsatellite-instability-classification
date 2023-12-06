from src.configs import Configs
from src.components.objects.Logger import Logger
import pandas as pd
import os
from src.training_utils import init_training_transforms, init_training_callbacks
from src.components.models.VariantClassifier import VariantClassifier
import torch


coad_filenames = ['TCGA-AA-A00K-01Z-00-DX1.25BD5724-7D30-4885-A3F9-D94FAED92984.svs', 'TCGA-AA-3848-01Z-00-DX1.bb018b1c-7748-4865-b00f-42edc35b5047.svs', 'TCGA-AA-A00A-01Z-00-DX1.B51DF257-6110-404C-9B20-A3C3453788F6.svs', 'TCGA-AA-A02K-01Z-00-DX1.732DD8F9-A21A-4E97-A779-3400A6C3D19D.svs', 'TCGA-AA-3971-01Z-00-DX1.348b1114-a9d5-4a37-8dfd-3bc8738fec35.svs', 'TCGA-D5-6922-01Z-00-DX1.6c11a531-71a3-45ff-b01a-49719b6a195c.svs', 'TCGA-AA-3856-01Z-00-DX1.973974e7-fcfe-4866-bc0c-50645c6c304b.svs', 'TCGA-AD-5900-01Z-00-DX1.ff1dbf00-d9c5-45a2-9732-07b46f4e1471.svs', 'TCGA-A6-5666-01Z-00-DX1.6f2cf971-edcb-415d-9709-feb7422cddc4.svs', 'TCGA-AA-3527-01Z-00-DX1.20035f6a-71f8-4d1f-a356-64001be9c2dd.svs', 'TCGA-AM-5820-01Z-00-DX1.365538bf-65ff-4fcd-8862-90627484431c.svs', 'TCGA-CK-5916-01Z-00-DX1.726a78b1-e64f-4dd6-8f7e-e43e98f1f453.svs', 'TCGA-CK-4951-01Z-00-DX1.abdbb15c-fd40-4a55-bf54-5668b3d4ea13.svs', 'TCGA-D5-6926-01Z-00-DX1.3830423a-3587-432b-9a6c-84f838e49fe6.svs', 'TCGA-AD-6895-01Z-00-DX1.7FB1FBC6-683B-4285-89D1-A7A20F07A9D4.svs', 'TCGA-G4-6306-01Z-00-DX1.962227ca-b0d6-4cf4-afea-8f7c2f9b2477.svs', 'TCGA-AY-4071-01Z-00-DX1.6C426E14-2DBD-4009-A6B5-B8B7B8F5888C.svs', 'TCGA-AA-3655-01Z-00-DX1.D78D8DBE-E74F-491D-AC9C-10E4C4E7BB02.svs', 'TCGA-G4-6628-01Z-00-DX1.d67973d1-9544-47e1-9ecb-e9d8d7f310e6.svs', 'TCGA-AY-6386-01Z-00-DX1.2B976983-5724-4335-8B47-9B44271B0A01.svs', 'TCGA-A6-6649-01Z-00-DX1.9439bce8-5715-4d76-a5d8-c6cbe1b79435.svs', 'TCGA-AA-3869-01Z-00-DX1.ef8b8cf0-5de5-4475-ac37-286d58604a0f.svs', 'TCGA-AA-3675-01Z-00-DX1.9afbbb26-2574-46af-8154-5f39bab6f01a.svs', 'TCGA-G4-6294-01Z-00-DX1.29a1716d-8875-49f0-8a83-f416221623b3.svs', 'TCGA-AZ-4308-01Z-00-DX1.804e054d-e206-4b34-a992-86317ef515d9.svs', 'TCGA-CK-5913-01Z-00-DX1.64d267c5-895f-4274-9d98-bfd2d338aee2.svs', 'TCGA-AA-A00L-01Z-00-DX1.4F57C465-BC1C-4774-B003-F7F29C6A69D0.svs', 'TCGA-AA-3712-01Z-00-DX1.00E0ACE2-8CC5-4063-9C65-3CDD7F21B189.svs', 'TCGA-G4-6626-01Z-00-DX1.20e7691e-9ef5-4278-a7d4-3967e36f24d5.svs', 'TCGA-AA-3858-01Z-00-DX1.6336815f-9887-4f74-a15d-78e7f6cacb59.svs', 'TCGA-A6-2671-01Z-00-DX1.13d1a0d9-78cd-4cfc-b670-34a79ebe52ee.svs', 'TCGA-AA-3812-01Z-00-DX1.c501fc71-8370-4034-b32a-1bb7cd846881.svs', 'TCGA-CA-5797-01Z-00-DX1.6549a80e-4b68-4147-949b-6149ab680313.svs', 'TCGA-DM-A28F-01Z-00-DX1.3ebf56a0-8f55-4681-bc7f-1e20d802a966.svs', 'TCGA-D5-6930-01Z-00-DX1.fbf9468b-67c6-413d-a188-707ee2ab9b95.svs', 'TCGA-D5-5540-01Z-00-DX1.4e4c69ca-f736-4db3-a401-c4f814d975dd.svs', 'TCGA-AA-A02E-01Z-00-DX1.04D47621-9DCF-437C-A4D6-44D17579FE6D.svs', 'TCGA-AA-3510-01Z-00-DX1.B4FCE76A-3B60-4D7D-9F3E-973AE17EA8E2.svs', 'TCGA-A6-6137-01Z-00-DX1.f50ab63c-05b0-49ea-9ceb-ed10cd6dc027.svs', 'TCGA-AY-6196-01Z-00-DX1.747B47B0-386A-4BA9-A8B7-274F1699D15E.svs', 'TCGA-F4-6461-01Z-00-DX1.f7da24ac-4a3c-4814-8d11-3138a954e0df.svs', 'TCGA-AA-3560-01Z-00-DX1.7ca786bd-777e-4b33-8778-fc5e2b061ff1.svs', 'TCGA-CM-5341-01Z-00-DX1.af4f75ff-3971-4639-8ef4-918ef4b29df0.svs', 'TCGA-AA-A01X-01Z-00-DX1.7433F54C-2A79-467A-8FEA-638AE48F42A0.svs', 'TCGA-AA-3561-01Z-00-DX1.1b5a2925-53f9-470f-a62c-cc2e5d5abb58.svs', 'TCGA-CM-5868-01Z-00-DX1.70f2e193-248d-4bf9-a875-49c314223f70.svs', 'TCGA-A6-3808-01Z-00-DX1.2b5a7ba3-133c-48be-87c6-199c4af208a0.svs', 'TCGA-A6-6138-01Z-00-DX1.11a4fad6-dfa3-4236-b714-bf1023b54622.svs', 'TCGA-AD-6963-01Z-00-DX1.7df2e133-5f24-4c0a-b7f5-5a65fe3420c9.svs', 'TCGA-AA-A01Q-01Z-00-DX1.4432694B-F24B-4942-91FD-27DEF1D84921.svs', 'TCGA-G4-6307-01Z-00-DX1.fff548a0-8bc8-428e-a4ce-3a5d0b3f060b.svs', 'TCGA-A6-3807-01Z-00-DX1.c3de2064-4f8d-4786-9ff9-2e0f44293717.svs', 'TCGA-AA-3688-01Z-00-DX1.642ce194-6dc0-4a96-aa79-674f48966df3.svs', 'TCGA-CM-6161-01Z-00-DX1.552104aa-6fd7-4d53-918b-fe67d359815c.svs', 'TCGA-AA-A03F-01Z-00-DX1.8E1A83FE-2C85-4444-A8FC-D0691817968A.svs', 'TCGA-CA-5254-01Z-00-DX1.cd986205-1db0-457b-9a28-75bed56376cb.svs', 'TCGA-D5-6541-01Z-00-DX1.b342c06b-8c59-4218-82f5-388568037e41.svs', 'TCGA-CM-6676-01Z-00-DX1.dcc2bf23-ecaa-4952-8485-fc609af66298.svs', 'TCGA-QG-A5YX-01Z-00-DX1.28125B5A-B696-44AE-8A86-72E2CF7B9A6A.svs', 'TCGA-F4-6808-01Z-00-DX1.c5c94635-21de-4edb-a903-4a2c914a5795.svs', 'TCGA-AA-3844-01Z-00-DX1.bf88ce1f-0601-40c8-813e-4e3df51bd2f0.svs', 'TCGA-AU-6004-01Z-00-DX1.12a234b3-1843-47ca-b650-37c1a631f489.svs', 'TCGA-A6-5667-01Z-00-DX1.1973b80d-b6b8-4ed8-9bc1-3aef51fbd9e6.svs', 'TCGA-AZ-4614-01Z-00-DX1.c1488dfe-528f-4dd4-b5f8-da81fbb4ec0b.svs', 'TCGA-AA-3555-01Z-00-DX1.d471efa5-7638-43e7-a2cd-93c1aed135d0.svs', 'TCGA-AA-3693-01Z-00-DX1.0e18d1db-cdff-433b-a150-6c759c4dc3bb.svs', 'TCGA-AA-3977-01Z-00-DX1.08ffa326-08fd-4215-9bf7-81fcf33b4f5a.svs', 'TCGA-F4-6460-01Z-00-DX1.92a182ea-f22a-4d74-bfb6-34d3cd757dce.svs', 'TCGA-AA-3980-01Z-00-DX1.93383cb9-59a7-431d-b268-3c3d59a1120e.svs', 'TCGA-AA-3548-01Z-00-DX1.41949ab5-79f2-4729-9d54-c0fca1daf124.svs', 'TCGA-A6-2678-01Z-00-DX1.bded5c5c-555a-492a-91c7-151492d0ee5e.svs', 'TCGA-AA-A00O-01Z-00-DX1.6787693F-6E3E-479A-A4DE-44186659285F.svs', 'TCGA-G4-6304-01Z-00-DX1.cf678a4f-5448-4d6a-a36f-cd1aec4d48a7.svs', 'TCGA-AA-3956-01Z-00-DX1.9438cc23-9537-424c-9a70-852919548387.svs', 'TCGA-A6-2681-01Z-00-DX1.5e11f090-a19d-4d5c-bcf6-c219b55d02bc.svs', 'TCGA-AA-3841-01Z-00-DX1.10f98d09-027f-4495-9ad8-7c8dc49a07d7.svs', 'TCGA-CM-4752-01Z-00-DX1.ac26d5ca-f554-4766-a4c3-f90a8c327dd4.svs', 'TCGA-WS-AB45-01Z-00-DX1.1FD99E7A-830F-40DC-98CD-53C62C678AC6.svs', 'TCGA-DM-A280-01Z-00-DX1.e7cfcec1-e284-4b94-80c2-cbb3186f7c6f.svs', 'TCGA-AA-3842-01Z-00-DX1.8bbbd702-2b17-4c3e-a8bd-55c3ae8aaba3.svs', 'TCGA-DM-A1D4-01Z-00-DX1.38346604-2BAF-44F8-BD96-5BF58253C6AD.svs', 'TCGA-AA-3509-01Z-00-DX1.EAE46823-3132-486F-8C2A-C0F548A08335.svs', 'TCGA-T9-A92H-01Z-00-DX1.9BA130C5-CAEF-4452-BB1F-61963B0DA3C5.svs', 'TCGA-CA-6716-01Z-00-DX1.fd53102c-7f2b-47f1-b4d1-5693e81a0478.svs', 'TCGA-AZ-6605-01Z-00-DX1.4d634c3e-a1c0-42e1-a4d6-5980eddfa0ca.svs', 'TCGA-F4-6807-01Z-00-DX1.84bfb631-af3d-45e7-a7db-730844a53625.svs', 'TCGA-A6-5662-01Z-00-DX1.82569684-1c31-4346-af9b-c296a020f624.svs', 'TCGA-4T-AA8H-01Z-00-DX1.A46C759C-74A2-4724-B6B5-DECA0D16E029.svs', 'TCGA-CM-6163-01Z-00-DX1.012a7433-73bb-4584-957b-f92c8877a114.svs', 'TCGA-D5-6538-01Z-00-DX1.fab8da8e-1e0a-4fb2-987c-9792c05d5a3a.svs', 'TCGA-AA-3532-01Z-00-DX1.00c7a378-a7c5-4fb4-9f53-6197be81c2eb.svs', 'TCGA-AA-3696-01Z-00-DX1.947f2c09-dfe9-4fdb-bf1a-9bf46d67f617.svs', 'TCGA-G4-6298-01Z-00-DX1.83055d52-71f7-46ec-be53-11d86b19b4cf.svs', 'TCGA-AA-3681-01Z-00-DX1.576342cf-0f40-404a-b3c5-b33103f86777.svs', 'TCGA-G4-6314-01Z-00-DX1.bea21980-9584-4382-9de3-4c5114edb10d.svs', 'TCGA-AA-3521-01Z-00-DX1.9d6be975-7be1-4f6c-99db-5101369c6624.svs', 'TCGA-CM-5862-01Z-00-DX1.df57752c-5937-40f2-a48f-37a147a82139.svs', 'TCGA-CM-5861-01Z-00-DX1.b900abc0-ecca-48e1-98ba-fbc99a6dae3e.svs', 'TCGA-AY-A69D-01Z-00-DX1.94582A46-7470-4265-8371-23BC246431EE.svs', 'TCGA-DM-A28H-01Z-00-DX1.daf607db-57d0-4685-8dd1-f6d0a9ee0435.svs', 'TCGA-AA-3979-01Z-00-DX1.e63b4db2-dc9b-4afb-a288-89a905beacd0.svs', 'TCGA-AA-3558-01Z-00-DX1.c4af1f52-2e81-4d66-9561-ce446dcace03.svs', 'TCGA-D5-6923-01Z-00-DX1.ad5211f6-32a3-42c6-8594-93cb4432b5f6.svs', 'TCGA-G4-6322-01Z-00-DX1.8676af67-716a-4052-a5e7-024b2e29c492.svs', 'TCGA-D5-6529-01Z-00-DX1.3b2ef23a-29b5-4a22-893c-6114d8244e68.svs', 'TCGA-AA-3966-01Z-00-DX1.7bc0c76e-f2d9-4abd-b63c-dad01aa4b1f7.svs', 'TCGA-QG-A5YV-01Z-00-DX1.9B7FD3EA-D1AB-44B3-B728-820939EF56EA.svs', 'TCGA-AZ-4315-01Z-00-DX1.1a2c2771-3e59-47c3-b380-42110c545e6b.svs', 'TCGA-AA-3534-01Z-00-DX1.a49495e0-93b2-41fa-9229-6375533578b5.svs', 'TCGA-CA-6718-01Z-00-DX1.9774472f-a29a-4b2b-8e50-ccbf9e5f9686.svs', 'TCGA-A6-5657-01Z-00-DX1.d0cab3dd-8758-4a3e-8bb2-7cd9411dbeb6.svs', 'TCGA-AA-A01D-01Z-00-DX1.A6FE424C-9BF8-4605-8A78-7BD7B83BEC61.svs', 'TCGA-AD-A5EK-01Z-00-DX1.709B3557-3E24-4CAB-8CD6-604C9438BC12.svs', 'TCGA-AA-3549-01Z-00-DX1.2fe99d54-c61b-4867-bafe-efe4f291c429.svs', 'TCGA-D5-6928-01Z-00-DX1.f8a8fb91-c23e-418e-b9a3-500af1402ce1.svs', 'TCGA-D5-6539-01Z-00-DX1.fe2a2e60-1db0-4019-9920-99416b34f05e.svs', 'TCGA-AA-3846-01Z-00-DX1.d6233d91-4d33-424a-99c1-8173fbeb5090.svs', 'TCGA-D5-6536-01Z-00-DX1.a4528e3f-770e-4271-9943-d3a8b8bd3e9d.svs', 'TCGA-AA-A01T-01Z-00-DX1.0C795296-87D6-4B90-9363-CF5CD7A2924D.svs', 'TCGA-AA-A03J-01Z-00-DX1.4E57E86E-ADEE-4837-9F91-E9CA141F7ACC.svs', 'TCGA-AA-3814-01Z-00-DX1.15a569dc-30d6-4bef-908b-6183df4e9e01.svs', 'TCGA-F4-6570-01Z-00-DX1.57a15bf3-d9a1-4da7-b71a-7b992a275bdf.svs', 'TCGA-A6-2680-01Z-00-DX1.7b77c0fb-f51d-4d16-ae77-f7615b1d0b87.svs', 'TCGA-DM-A28G-01Z-00-DX1.5e8602bd-31e1-4813-8214-cd56280defe5.svs', 'TCGA-A6-2683-01Z-00-DX1.0dfc5d0a-68f4-45e1-a879-0428313c6dbc.svs', 'TCGA-DM-A28K-01Z-00-DX1.766edac4-a5c8-45ef-aff6-73e308c1f442.svs', 'TCGA-AA-3531-01Z-00-DX1.19cdaa4b-5a53-4198-90da-5800827d90bf.svs', 'TCGA-CM-6172-01Z-00-DX1.a5d23c88-a173-46a2-b8dd-6d873b8216c7.svs', 'TCGA-A6-2674-01Z-00-DX1.d301f1f5-6f4a-49e6-9c93-f4e8b7f616b8.svs', 'TCGA-AA-3877-01Z-00-DX1.36902310-bc0b-4437-9f86-6df85703e0ad.svs', 'TCGA-NH-A6GA-01Z-00-DX1.33AFBF24-84BD-4E21-8A2D-A565AD3E4376.svs', 'TCGA-AA-3520-01Z-00-DX1.3e2b41c2-eb21-4f68-8946-92b59cc1f969.svs', 'TCGA-AM-5821-01Z-00-DX1.0851bd43-6c23-4db3-a50a-7c9fb5080150.svs', 'TCGA-AA-A02H-01Z-00-DX1.5343879F-6C5D-48B3-8D78-D895ED118F42.svs', 'TCGA-D5-6929-01Z-00-DX1.5e555bba-87b2-440c-b6d4-e6fec3f7bf3a.svs', 'TCGA-AY-A8YK-01Z-00-DX1.89E3C546-0425-449B-A6FB-1C35168EA7EB.svs', 'TCGA-CM-4750-01Z-00-DX1.250ea50f-3aae-4fcd-9ba9-25cf2115525f.svs', 'TCGA-A6-6142-01Z-00-DX1.e923ce20-d3c3-4d21-9e7c-d999a3742f9b.svs', 'TCGA-AA-3864-01Z-00-DX1.f6992bc7-ba05-4c30-9500-8f7b07b30f9a.svs', 'TCGA-AA-A004-01Z-00-DX1.2576461E-7FA3-4CC6-8CC3-58D8E88CE04D.svs', 'TCGA-AA-3529-01Z-00-DX1.99453fef-afe8-4a43-a64f-df2d48ef9e55.svs', 'TCGA-NH-A5IV-01Z-00-DX1.1A6F9F12-C00E-47F2-9400-541BA230EFBE.svs', 'TCGA-NH-A50V-01Z-00-DX1.408BA0A6-E569-4464-A8CB-D6553A4DF9E0.svs', 'TCGA-AA-3950-01Z-00-DX1.2a81cf11-4c16-4e9e-8809-6f63152060da.svs', 'TCGA-AA-3667-01Z-00-DX1.28dc3612-1c43-4727-a134-698cc4315dc3.svs', 'TCGA-AZ-6606-01Z-00-DX1.aa79b3d8-8ff1-4171-96dc-94bc7d073d93.svs', 'TCGA-AD-6965-01Z-00-DX1.0330727c-42f5-4a08-a35d-af81eda1d0f1.svs', 'TCGA-AA-3516-01Z-00-DX1.0a9d9207-6dc2-44b9-89ea-16418430c484.svs', 'TCGA-NH-A6GB-01Z-00-DX1.AD90C375-54ED-4EE4-A537-59A2E3FE4BCD.svs', 'TCGA-CM-4751-01Z-00-DX1.F72E1883-5293-4351-A8DC-C4EA5D8F797C.svs', 'TCGA-CM-6171-01Z-00-DX1.74d4391e-3dbc-4ad4-b188-3b11ac65e6d8.svs', 'TCGA-F4-6805-01Z-00-DX1.9927edbb-2801-4988-b113-1fdfd31a72a0.svs', 'TCGA-AA-A00N-01Z-00-DX1.79D0A833-8411-486B-9BED-7B5E203D02F2.svs', 'TCGA-AU-3779-01Z-00-DX1.4134005A-8A79-46DC-8737-B3C8AAC2DFCA.svs', 'TCGA-AA-3544-01Z-00-DX1.96850cbf-2305-4b65-8f06-db801af51cc3.svs', 'TCGA-CM-6680-01Z-00-DX1.68fe763e-d8f4-44ca-8604-4da4e57cee06.svs', 'TCGA-CM-6675-01Z-00-DX1.4f2301e2-2894-484d-8a52-7be902a9861b.svs', 'TCGA-AA-3660-01Z-00-DX1.CCD0F50D-9991-4CC2-AC77-AD1F78D8CFEB.svs', 'TCGA-DM-A1DA-01Z-00-DX1.00001FEF-3B63-4C6F-952A-1D5F6F51CD22.svs', 'TCGA-AA-3556-01Z-00-DX1.63a74b91-44e8-4ffd-8737-bcf6992183c3.svs', 'TCGA-AA-3710-01Z-00-DX1.78082263-b7e9-4281-aa55-2da5f80e4499.svs', 'TCGA-D5-6533-01Z-00-DX1.a4b5096e-88cc-4797-b8d5-1a9cf1e74a55.svs', 'TCGA-AA-A00W-01Z-00-DX1.24770462-BD63-4881-9AE3-9198E9093AD9.svs', 'TCGA-DM-A1D7-01Z-00-DX1.4F3CF25D-A350-4A92-A891-7FFE40BE2710.svs', 'TCGA-D5-6920-01Z-00-DX1.e184673f-e7e9-44aa-9dae-7054bd1d0d00.svs', 'TCGA-AZ-4616-01Z-00-DX1.0a0f6eaa-4db6-4479-a9df-f09387f555b1.svs', 'TCGA-AA-3685-01Z-00-DX1.57ef312b-70e0-46f5-b847-0e0ac32f1824.svs', 'TCGA-DM-A282-01Z-00-DX1.65f620ea-37be-4d3c-a993-a2bfb552108c.svs', 'TCGA-AZ-6608-01Z-00-DX1.40d9f93f-f7d8-4138-9af1-bb579c53194b.svs', 'TCGA-D5-6530-01Z-00-DX1.5c4bbcd1-51ba-467d-93f9-f2a9e7c5e010.svs', 'TCGA-AA-3984-01Z-00-DX1.d0cb7571-c612-4410-ac03-ebe800ad6767.svs', 'TCGA-G4-6295-01Z-00-DX1.9e7ae22f-daac-42cb-a879-bcf505d1c725.svs', 'TCGA-AD-6890-01Z-00-DX1.4778042f-f210-489c-bb76-b4fe16b0d500.svs', 'TCGA-AD-6899-01Z-00-DX1.646f5e1a-212f-4b15-8689-8b55f7ba8c47.svs', 'TCGA-AA-A02F-01Z-00-DX1.6E214530-87AE-4E9F-89A9-E35BA9C69BB0.svs', 'TCGA-DM-A1D6-01Z-00-DX1.BCDA4D6C-8424-477D-9FAF-907206D2DDD6.svs', 'TCGA-F4-6463-01Z-00-DX1.a3fa6fb4-ce9d-4f0d-b5f7-3c9da7322cd0.svs', 'TCGA-F4-6809-01Z-00-DX1.5ab8333f-0c77-4685-8701-4130a93e6f3a.svs', 'TCGA-D5-6531-01Z-00-DX1.32241731-5890-424e-96d5-b897e770f03c.svs', 'TCGA-D5-6537-01Z-00-DX1.f81ccf91-7ce6-4ccc-8278-cc05f639aca7.svs', 'TCGA-AA-3973-01Z-00-DX1.05cee752-3f4e-442d-a093-dcfb2b6130f0.svs', 'TCGA-AY-6197-01Z-00-DX1.AD42F96E-6583-4AB8-A6BD-C8334EA9DE14.svs', 'TCGA-A6-5656-01Z-00-DX1.8a8ebf52-8217-4288-8886-7eefa6cdfdca.svs', 'TCGA-CM-6162-01Z-00-DX1.806a99a3-cda2-4dde-8d13-d22912b44d49.svs', 'TCGA-A6-5664-01Z-00-DX1.622f6650-1926-4fa2-b42b-74122d9a68a4.svs', 'TCGA-AA-3680-01Z-00-DX1.9eef1b8f-c3c1-486f-83e7-a88182ce892a.svs', 'TCGA-CM-6165-01Z-00-DX1.d59faefa-f647-4617-9dcf-4fd6ab45b4e6.svs', 'TCGA-AA-3818-01Z-00-DX1.80d3eeeb-9a4d-4211-90e8-605a4b809a63.svs', 'TCGA-G4-6321-01Z-00-DX1.20bd4687-4b24-4666-a722-d42b9731136e.svs', 'TCGA-G4-6309-01Z-00-DX1.3eb20fdc-eb86-4bd5-8194-76b02d4fa472.svs', 'TCGA-AZ-4682-01Z-00-DX1.abca2345-ed4a-4f64-af1d-0e60f81b1288.svs', 'TCGA-A6-2685-01Z-00-DX1.c69e23f4-34c9-41ff-a037-44bf7bbf33cd.svs', 'TCGA-AZ-6599-01Z-00-DX1.9d6aae3d-6934-4e96-8699-db41e1194f29.svs', 'TCGA-A6-A5ZU-01Z-00-DX1.8E7A136C-46C4-4233-A747-EBDC4F3227FB.svs', 'TCGA-A6-A5ZU-01Z-00-DX2.221EC8DE-4029-4ED2-8D84-95647BD39E03.svs', 'TCGA-AA-3875-01Z-00-DX1.016712a0-4226-4086-857c-3d6d85f186e3.svs', 'TCGA-CM-6674-01Z-00-DX1.4a08b16a-788e-43dc-85d2-baff6e911de2.svs', 'TCGA-AA-3713-01Z-00-DX1.8148ACEB-7C1E-4D29-B908-F3729657EA4F.svs', 'TCGA-AA-A02O-01Z-00-DX1.CB9BE08B-D78B-46B3-8339-A3DADD24439F.svs', 'TCGA-AZ-4681-01Z-00-DX1.e468c1a1-e251-4521-82fe-526c9c5f8190.svs', 'TCGA-AZ-6607-01Z-00-DX1.b0a25161-7e13-42d6-9271-ecc5ecce2232.svs', 'TCGA-AA-A022-01Z-00-DX1.2673F279-5DF6-4E71-92B2-A589DD8F583B.svs', 'TCGA-AA-3854-01Z-00-DX1.1564d865-6653-4be1-951e-ea9fab0102a7.svs', 'TCGA-CA-5796-01Z-00-DX1.88141789-4240-4ab7-8db1-e4cb7ee1ebda.svs', 'TCGA-AA-A00Q-01Z-00-DX1.D427F78B-6640-400B-B8F8-B5568B1C4321.svs', 'TCGA-AA-3488-01Z-00-DX1.EDF60198-F7AB-45BB-9A1B-C2E2FA141989.svs', 'TCGA-AZ-6601-01Z-00-DX1.40681471-3104-48be-8b57-55dba1f432f8.svs', 'TCGA-AA-A02R-01Z-00-DX1.B332C84C-EE97-4855-B773-9B5CBFA45096.svs', 'TCGA-AA-3970-01Z-00-DX1.712c069a-aeaf-498b-80fa-7bb481b13825.svs', 'TCGA-AA-3986-01Z-00-DX1.db60e495-c0eb-416c-b65b-55ce62ed10b0.svs', 'TCGA-CK-4952-01Z-00-DX1.0e98c7b4-5f80-485d-a191-ad93564b5f96.svs', 'TCGA-AA-3862-01Z-00-DX1.67a0bc0d-1fe0-4c90-bb2d-5b12224cc846.svs', 'TCGA-AA-3496-01Z-00-DX1.B109A6F3-02E0-4181-B69A-00CBA758C074.svs', 'TCGA-AA-3494-01Z-00-DX1.E275AF20-3AA7-4191-BD1F-FFE744CA6A2F.svs', 'TCGA-CM-5864-01Z-00-DX1.2cb87875-6cae-4d8e-9c93-4a83941c0ca9.svs', 'TCGA-AA-3955-01Z-00-DX1.e0eea910-db79-4797-b9db-bb8bfe35306d.svs', 'TCGA-A6-6653-01Z-00-DX1.e130666d-2681-4382-9e7a-4a4d27cb77a4.svs', 'TCGA-AA-A00F-01Z-00-DX1.7E748515-2D18-4061-AF9A-E1446E44E7B8.svs', 'TCGA-CK-5914-01Z-00-DX1.dfa6d814-6ddb-4058-a236-d57303cbfbe9.svs', 'TCGA-AY-4070-01Z-00-DX1.dd650ac6-8480-4fd8-85b8-15a7840a5933.svs', 'TCGA-AA-3852-01Z-00-DX1.d662015f-398d-4a98-b384-46221070da2f.svs', 'TCGA-A6-6648-01Z-00-DX1.88b9a490-0bed-43f3-bd74-1bf2810f6884.svs', 'TCGA-AA-3947-01Z-00-DX1.77005e19-8a8e-4b82-89e9-c81af9b41193.svs', 'TCGA-AA-3673-01Z-00-DX1.a80676fa-5481-4b63-9639-dbeb31ae82d8.svs', 'TCGA-G4-6302-01Z-00-DX1.78bd777d-4b82-44de-9ef1-c4c641364015.svs', 'TCGA-AA-3866-01Z-00-DX1.f93457c3-abaa-4268-84e2-394d7c1aa523.svs', 'TCGA-A6-5661-01Z-00-DX1.bad2d858-11b4-4b9c-a720-daaae592cf48.svs', 'TCGA-A6-2684-01Z-00-DX1.be127778-e160-4ae3-9e5a-13a16eae2e7a.svs', 'TCGA-A6-5665-01Z-00-DX1.3ad2c249-d138-4037-a59b-4747ce2b789a.svs', 'TCGA-DM-A1HA-01Z-00-DX1.E56FC26A-DDB9-4121-9E79-5009FB23CCEB.svs', 'TCGA-NH-A6GC-01Z-00-DX1.29073D7E-5EEF-4BBA-96BE-DC8C69924C42.svs', 'TCGA-CM-4747-01Z-00-DX1.e0fac451-1322-464e-b718-174e9db33f39.svs', 'TCGA-CK-4948-01Z-00-DX1.cd6ecbec-9136-4ce7-9a96-eb1ac975b30f.svs', 'TCGA-G4-6297-01Z-00-DX1.3e37fa8b-c10e-4e44-9933-5bcbbe088fe0.svs', 'TCGA-AA-A01G-01Z-00-DX1.8A288E53-BA38-4BAC-81B5-2E0E41EA0D85.svs', 'TCGA-AA-A02W-01Z-00-DX1.3D9DD408-C389-411D-B4AC-6DC531D35BAD.svs', 'TCGA-AA-A02J-01Z-00-DX1.1326204B-9264-482C-9F75-795DD085C0DF.svs', 'TCGA-CK-4947-01Z-00-DX1.b257d5fd-b97b-4987-b088-77f044ca7fe2.svs', 'TCGA-AA-A01I-01Z-00-DX1.D24F43B2-F46E-4F7F-85A0-91F3A04E0785.svs', 'TCGA-A6-A56B-01Z-00-DX1.52FE9FA5-05F1-49EA-98BE-887CF7B3A52F.svs', 'TCGA-A6-A566-01Z-00-DX1.325BC1B7-2D0D-43CC-A23B-7D13B2DF665D.svs', 'TCGA-A6-6654-01Z-00-DX1.ed491b61-7c44-4275-879b-22f8007b5ff1.svs', 'TCGA-CA-6715-01Z-00-DX1.d5db8085-f91a-4eee-b15f-61960af713af.svs', 'TCGA-AA-3967-01Z-00-DX1.b80c87c9-00f4-44f6-bc59-19d2b94942ac.svs', 'TCGA-A6-2676-01Z-00-DX1.c465f6e0-b47c-48e9-bdb1-67077bb16c67.svs', 'TCGA-F4-6459-01Z-00-DX1.80a78213-1137-4521-9d60-ac64813dec4c.svs', 'TCGA-D5-6535-01Z-00-DX1.0d7485ff-cf98-4c86-8a61-c7364f41b8b0.svs', 'TCGA-AA-A01C-01Z-00-DX1.C69C8FC3-04A2-46B8-8577-7A7F082248CB.svs', 'TCGA-QG-A5Z2-01Z-00-DX1.51896C31-235E-48EF-90F7-FC05350CA564.svs', 'TCGA-QG-A5Z2-01Z-00-DX2.F2352352-8F00-4BB3-8A62-8D1C1E374F95.svs', 'TCGA-AA-A01S-01Z-00-DX1.1F2812C1-9807-4D14-8071-3FE15236EB44.svs', 'TCGA-AA-3939-01Z-00-DX1.6ceb6e8f-a469-4f42-9597-8bf853d95640.svs', 'TCGA-AA-3994-01Z-00-DX1.ca18c0cb-88b4-4a31-be1f-cca57dfadabc.svs', 'TCGA-CM-6169-01Z-00-DX1.0381c243-02b8-4f1d-840c-19ef44d4b92c.svs', 'TCGA-AD-6889-01Z-00-DX1.5269A81E-5391-4875-8F2A-6505BC5BBFD9.svs', 'TCGA-G4-6588-01Z-00-DX1.0747172e-f630-4b7c-9341-55078585ae00.svs', 'TCGA-AA-3524-01Z-00-DX1.b1aae264-87be-4514-8f9d-25660b39caa7.svs', 'TCGA-5M-AAT5-01Z-00-DX1.548E7CEB-48FB-4037-A616-39AB025E7A73.svs', 'TCGA-NH-A50T-01Z-00-DX1.4624B690-C0DE-42BD-852C-6EBABF40255F.svs', 'TCGA-A6-6141-01Z-00-DX1.34b5db5c-74df-47d9-bb89-beec93ded868.svs', 'TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4.svs', 'TCGA-AA-3982-01Z-00-DX1.2d2e4de6-5b8e-4fbb-a9a8-4fbb48b5492a.svs', 'TCGA-AA-A00J-01Z-00-DX1.BA85D337-6687-44A6-A8DD-0CE889134BA0.svs', 'TCGA-D5-6931-01Z-00-DX1.c1d00654-b5ff-4485-90a6-97ae9e7bd7fa.svs', 'TCGA-NH-A8F7-01Z-00-DX1.5CB8911D-07C3-4EF2-A97D-A62B441CF79E.svs', 'TCGA-CM-6164-01Z-00-DX1.ccf5ce96-b732-4c35-b177-d3dbe2ed89cb.svs', 'TCGA-AA-3861-01Z-00-DX1.1735d004-51bd-447a-add4-05f0c583c6ca.svs', 'TCGA-A6-6650-01Z-00-DX1.92f39e59-8784-4dfd-a06f-804bebcdfb26.svs', 'TCGA-G4-6586-01Z-00-DX1.f19ef98f-9540-4b8d-bd13-5891e79b2576.svs', 'TCGA-CM-6168-01Z-00-DX1.96af6eb2-9d51-4671-baf8-1a73d0c66869.svs', 'TCGA-CM-6679-01Z-00-DX1.b3a899df-256a-4546-9428-a6dd2695b2cf.svs', 'TCGA-AZ-5403-01Z-00-DX1.1c557fea-6627-48e9-abb9-79da22c40cef.svs', 'TCGA-T9-A92H-01Z-00-DX3.1DE7D5ED-60F7-4645-8243-AB0C027B3ED7.svs', 'TCGA-DM-A28E-01Z-00-DX1.4381ffe6-3918-4fdd-b192-000f2b737b22.svs', 'TCGA-AA-3492-01Z-00-DX1.32D79909-71D5-4843-847E-AECA5DBC963D.svs', 'TCGA-AA-3543-01Z-00-DX1.20129c52-157d-4d66-809f-d21694683c8d.svs', 'TCGA-AA-3941-01Z-00-DX1.7c445fb8-53ee-4813-9ca8-8f7c3cc0bdde.svs', 'TCGA-CK-6747-01Z-00-DX1.7824596c-84db-4bee-b149-cd8f617c285f.svs', 'TCGA-AA-3526-01Z-00-DX1.82876320-2866-4ffa-81d7-3278f7150fc3.svs', 'TCGA-A6-3809-01Z-00-DX1.c26f03e8-c285-4a66-925d-ae9cba17d7b3.svs', 'TCGA-G4-6311-01Z-00-DX1.f1b98598-dbd8-4ba5-9ec7-5c93ccc82c81.svs', 'TCGA-DM-A1D8-01Z-00-DX1.2DD544F5-D72F-4840-B2D3-F361E032EA3B.svs', 'TCGA-AA-3664-01Z-00-DX1.bd07e7ef-0acb-43d8-a4f6-15b3442d2ed5.svs', 'TCGA-AA-3511-01Z-00-DX1.F66F89C7-147D-4EE9-A482-61C3033EF443.svs', 'TCGA-AA-A01K-01Z-00-DX1.2E147232-BC3C-48CC-B75E-43E6AA4A0BF8.svs', 'TCGA-A6-A567-01Z-00-DX1.F941874E-9BF7-4E8B-908C-41A638D62275.svs', 'TCGA-T9-A92H-01Z-00-DX2.43894C88-2096-4932-9E9D-17BDCACF988C.svs', 'TCGA-A6-2682-01Z-00-DX1.be71dca0-b9b7-40be-a6c6-9d053c7886a6.svs', 'TCGA-4N-A93T-01Z-00-DX1.82E240B1-22C3-46E3-891F-0DCE35C43F8B.svs', 'TCGA-4N-A93T-01Z-00-DX2.875E7F95-A6D4-4BEB-A331-F9D8080898C2.svs', 'TCGA-A6-4105-01Z-00-DX1.228b02a5-04fa-4392-bf03-b297c19665c3.svs', 'TCGA-AA-A029-01Z-00-DX1.36BA3129-431D-4AE5-98E6-BA064D0B5062.svs', 'TCGA-CK-6748-01Z-00-DX1.1dd76660-7858-470c-a27b-36586b788125.svs', 'TCGA-DM-A285-01Z-00-DX1.219e2829-8ffd-4b51-adce-cfd48293191b.svs', 'TCGA-D5-5541-01Z-00-DX1.2cd0a69e-879e-47aa-8035-1f9732ec4760.svs', 'TCGA-SS-A7HO-01Z-00-DX1.D20B9109-F984-40DE-A4F1-2DFC61002862.svs', 'TCGA-AA-3930-01Z-00-DX1.065c480c-9ac3-4d98-a351-cb320b6a5ba0.svs', 'TCGA-QG-A5YW-01Z-00-DX1.3242285F-FA82-4A92-9D0E-951013A3C91A.svs', 'TCGA-DM-A0XD-01Z-00-DX1.DAFA56D4-85CB-4FB1-B5BB-E993CA522FF8.svs', 'TCGA-AA-3972-01Z-00-DX1.a60c2c2b-71ea-4cd6-a56b-c8e409a181ac.svs', 'TCGA-A6-6652-01Z-00-DX1.30916007-088e-48bd-abf8-519f34e2c37a.svs', 'TCGA-CK-6746-01Z-00-DX1.0aae8eec-1e82-494b-8779-d79fea8bec0c.svs', 'TCGA-AZ-6603-01Z-00-DX1.e39d6a8c-a738-4d63-b094-11be49fac828.svs', 'TCGA-CA-5255-01Z-00-DX1.77310ae2-9c5f-48c4-9754-c5b30d287089.svs', 'TCGA-AD-6964-01Z-00-DX1.83AF88B9-C59B-48C6-A739-85ACB8F8ECA9.svs', 'TCGA-F4-6856-01Z-00-DX1.2872c7b5-b94d-4147-ad90-69f88668135a.svs', 'TCGA-AD-A5EJ-01Z-00-DX1.FA56CEAF-8B70-45EF-A2C9-8AA7BEB3D88A.svs', 'TCGA-CK-6751-01Z-00-DX1.df9e123a-c44c-4cc5-82de-ba7c4dbcb444.svs', 'TCGA-DM-A0X9-01Z-00-DX1.C7FC2C17-12CC-4F10-B54F-7C29379D834E.svs', 'TCGA-D5-6534-01Z-00-DX1.eb7b12b8-ad31-438f-8e1d-9bb76a560c86.svs', 'TCGA-AZ-4615-01Z-00-DX1.ecabbbb1-c1ed-4f60-b44f-b07eaa177208.svs', 'TCGA-CM-4748-01Z-00-DX1.e6307e86-29c5-4018-a94a-77fae9b08123.svs', 'TCGA-5M-AAT6-01Z-00-DX1.8834C952-14E3-4491-8156-52FC917BB014.svs', 'TCGA-DM-A28M-01Z-00-DX1.055b2d62-8a1e-4bdf-a49e-123ad0de657b.svs', 'TCGA-AA-A01Z-01Z-00-DX1.9724B55C-C5D9-4C8B-AA05-76C21BA1F046.svs', 'TCGA-AA-3554-01Z-00-DX1.53ea377e-6671-47bb-a2b6-b136d9686144.svs', 'TCGA-AA-3684-01Z-00-DX1.c6be6ea4-fa92-4499-b458-85c3a8b1e3b6.svs', 'TCGA-AA-A00R-01Z-00-DX1.7520405C-E7DD-46A4-BB68-ADFA511AEA64.svs', 'TCGA-AA-3538-01Z-00-DX1.60d0b039-25d6-4b71-a36f-5b2764a983ef.svs', 'TCGA-AA-3989-01Z-00-DX1.792D9A97-E06B-4F8C-8181-C4E5BD8B9A59.svs', 'TCGA-AA-3715-01Z-00-DX1.24d6e746-ad61-4587-a2b9-8903331b279c.svs', 'TCGA-A6-2677-01Z-00-DX1.dc0903dc-fef2-47ca-8f04-1ef25a4d8338.svs', 'TCGA-QL-A97D-01Z-00-DX1.6B48E95D-BE3C-4448-A1AF-6988C00B7AF1.svs', 'TCGA-A6-A565-01Z-00-DX1.42172A22-6F86-4661-BF2A-78815B721503.svs', 'TCGA-AA-3697-01Z-00-DX1.AAB8DB74-F76D-4D0A-A50E-E7F97504A3C4.svs', 'TCGA-D5-7000-01Z-00-DX1.fb08c430-2c8c-486b-a39d-7d28c5eae189.svs', 'TCGA-AA-3514-01Z-00-DX1.9e135da2-436e-47e3-9dbf-b2f577677828.svs', 'TCGA-D5-5537-01Z-00-DX1.14709d4c-eba0-48d0-87b8-5f34f74429d6.svs', 'TCGA-AA-A02Y-01Z-00-DX1.D1EE29E8-A27E-4035-807B-324A63239116.svs', 'TCGA-AA-3666-01Z-00-DX1.fe976853-fbde-46e8-b915-3c98440c9315.svs', 'TCGA-CM-6166-01Z-00-DX1.52eaa124-7ab5-4aaf-b074-7f89a4c53804.svs', 'TCGA-5M-AATE-01Z-00-DX1.483FFD2F-61A1-477E-8F94-157383803FC7.svs', 'TCGA-AA-3525-01Z-00-DX1.b6079f23-6ad2-41fb-885f-d7c68450c8d5.svs', 'TCGA-AA-A017-01Z-00-DX1.E7B1384E-3C57-4CE5-B85E-3B8FD328B5A2.svs', 'TCGA-AA-A00Z-01Z-00-DX1.47847702-E46E-40AA-9BA6-2ED1912D1E73.svs', 'TCGA-D5-6927-01Z-00-DX1.ff21d627-dbb8-4200-937b-f8be8b86b6d4.svs', 'TCGA-AZ-5407-01Z-00-DX1.5218a617-9817-44f4-8f00-8e9e3d04bd70.svs', 'TCGA-F4-6806-01Z-00-DX1.483d0fc7-220c-4b62-8c9e-a1004ce7450c.svs', 'TCGA-A6-4107-01Z-00-DX1.89bf3dd5-72a6-49cc-9857-df2c36884029.svs', 'TCGA-CM-6677-01Z-00-DX1.e3428c0a-a194-4e38-b105-8244701fcc71.svs', 'TCGA-QG-A5Z1-01Z-00-DX1.F3157C57-0F35-42D3-9CA5-C72D93F1BF89.svs', 'TCGA-QG-A5Z1-01Z-00-DX2.2CE72B6A-557F-43BD-BA4C-B252E14E46EF.svs', 'TCGA-AD-6901-01Z-00-DX1.0a69c0b5-6238-4c1a-bbbd-ea743bf6fc98.svs', 'TCGA-AA-3692-01Z-00-DX1.6e8c2370-54a7-4fce-b55c-bdb459828990.svs', 'TCGA-AA-3518-01Z-00-DX1.5aad7b19-2900-4f1c-9312-f8d8c4725449.svs', 'TCGA-AA-3831-01Z-00-DX1.0565c3fc-fe21-4b34-ae1b-626a46edaa9e.svs', 'TCGA-AA-3952-01Z-00-DX1.6a51a689-74cb-4204-9f7a-f5e3fc55fb2d.svs', 'TCGA-D5-6532-01Z-00-DX1.a28f2969-31ae-408f-99f5-5428e183e123.svs', 'TCGA-AA-3860-01Z-00-DX1.a63df9ca-6141-4bdc-8545-719fd9ae0aa5.svs', 'TCGA-CM-5860-01Z-00-DX1.95f23758-00b7-4602-b4ef-944130528f36.svs', 'TCGA-AZ-6600-01Z-00-DX1.9afe2f8f-bcfe-43df-a83b-6c183f226757.svs', 'TCGA-CM-6170-01Z-00-DX1.aa9c41ea-3894-4524-a94c-f44c6c53c2d0.svs', 'TCGA-D5-6924-01Z-00-DX1.a198456a-cf26-4cf3-a07a-edde8a4a710f.svs', 'TCGA-AA-3870-01Z-00-DX1.76e57cf5-6c8c-4b75-a8db-29d4522b66cb.svs', 'TCGA-CM-5863-01Z-00-DX1.2dceed07-9373-4103-be16-533dac9f283b.svs', 'TCGA-F4-6569-01Z-00-DX1.accbe317-9a4d-49b9-b9c9-4d2bb1301f67.svs', 'TCGA-CM-6167-01Z-00-DX1.7adf00e3-6768-46bb-814c-b2f04c472cc8.svs', 'TCGA-AA-3867-01Z-00-DX1.dbc11b4b-732c-4b0a-aaef-ba94b0218fe6.svs', 'TCGA-AY-A71X-01Z-00-DX1.68F9BC0F-1D60-4AEF-9083-509387038F03.svs', 'TCGA-G4-6315-01Z-00-DX1.2c3c17b0-c118-42b1-b1c9-7cc984e47f6c.svs', 'TCGA-AA-3522-01Z-00-DX1.bd54ca55-9036-4167-b8b9-14f4209b7e4d.svs', 'TCGA-AA-3968-01Z-00-DX1.54b76478-a822-49b5-8286-dcbbb2fba2f8.svs', 'TCGA-DM-A1D9-01Z-00-DX1.C286F663-142A-4F8E-BFCD-56E33F73F7E8.svs', 'TCGA-D5-6540-01Z-00-DX1.4ca9e502-959b-4fa8-a748-5fd0878e5c3f.svs', 'TCGA-AA-A01R-01Z-00-DX1.5D2BEC13-8F61-49D4-A96E-4C6C44BD5A38.svs', 'TCGA-DM-A28A-01Z-00-DX1.05b565c5-efa0-41be-a7e3-46f9166ddb7b.svs', 'TCGA-CK-5912-01Z-00-DX1.23a955f3-a1ed-4cb3-8e49-cbb3f789f3f5.svs', 'TCGA-AA-3845-01Z-00-DX1.20682536-a009-4184-a40b-cb889f37ad32.svs', 'TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs', 'TCGA-NH-A50U-01Z-00-DX1.CA30EE72-7149-44E0-9082-AC1922ADDB09.svs', 'TCGA-CK-5915-01Z-00-DX1.04650539-005e-4221-90d6-49706b1d7244.svs', 'TCGA-AA-3542-01Z-00-DX1.db284d9a-bde5-471c-ac37-5f3216d0f077.svs', 'TCGA-A6-5659-01Z-00-DX1.c671806f-013e-4d99-9841-cda5bd43eff1.svs', 'TCGA-CM-5344-01Z-00-DX1.586a3060-8c97-4619-b5b0-ad2d0d2b62cb.svs', 'TCGA-AA-3815-01Z-00-DX1.d6823390-73c1-431f-b480-0954f4df8224.svs', 'TCGA-AZ-6598-01Z-00-DX1.1fc4cd61-4524-413b-b36d-ad438785bc06.svs', 'TCGA-A6-3810-01Z-00-DX1.2940ca70-013a-4bc3-ad6a-cf4d9ffa77ce.svs', 'TCGA-CA-6719-01Z-00-DX1.590fea56-aae4-4108-9169-a67ec8cd95b7.svs', 'TCGA-A6-2672-01Z-00-DX1.e2a845c8-6d77-4120-9f43-abec84a66c1c.svs', 'TCGA-AA-3663-01Z-00-DX1.9AEDC003-2062-4876-8993-A5CEE4DDE1A9.svs', 'TCGA-CK-4950-01Z-00-DX1.03dcc4c2-2b63-45a2-8561-bf18193202b5.svs', 'TCGA-G4-6317-01Z-00-DX1.6521a551-1516-4431-b3d7-af0a46978bcf.svs', 'TCGA-AA-3489-01Z-00-DX1.AE299B70-B14C-4FFE-B1F9-38B2EB267FA9.svs', 'TCGA-AA-3517-01Z-00-DX1.dac0f9a3-fa10-42e7-acaf-e86fff0829d2.svs', 'TCGA-AA-3662-01Z-00-DX1.625F1BCC-5E59-411E-AE23-6F43CE6122B2.svs', 'TCGA-AY-A54L-01Z-00-DX1.BD4039B4-D732-418B-9CC9-064095A1F06F.svs', 'TCGA-AZ-4684-01Z-00-DX1.1c29deb2-b0e2-4788-a3e8-83ecab7f9208.svs', 'TCGA-CM-4744-01Z-00-DX1.527ead53-bd55-4321-adea-079bf5e2e8a5.svs', 'TCGA-AA-3672-01Z-00-DX1.6cc142eb-e77f-4c09-a6ac-e85470221812.svs', 'TCGA-CM-5348-01Z-00-DX1.2ad0b8f6-684a-41a7-b568-26e97675cce9.svs', 'TCGA-CM-6678-01Z-00-DX1.b0e06829-c119-4131-a2ec-22d41d8d6068.svs', 'TCGA-CM-4743-01Z-00-DX1.f54a6355-5623-498c-96b9-2ff1de6576c6.svs', 'TCGA-DM-A1DB-01Z-00-DX1.092D3ABD-7DFE-4193-B049-B3C3617706B0.svs', 'TCGA-AA-3679-01Z-00-DX1.b3445f8e-b143-4f24-9edd-8abdcb6b139b.svs', 'TCGA-AA-A024-01Z-00-DX1.5F24A31C-2F11-4768-9906-7BAB578C742D.svs', 'TCGA-AA-3553-01Z-00-DX1.45a24cd4-6eb2-4f20-9453-05476da34cdd.svs', 'TCGA-AA-3811-01Z-00-DX1.369d7946-911e-4e97-8ae7-32ef12b6b106.svs', 'TCGA-AA-3949-01Z-00-DX1.23748e80-0d7e-4238-8b29-f74cddae8596.svs', 'TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4.svs', 'TCGA-CM-4746-01Z-00-DX1.c83b9795-bf45-4100-9052-a7e485e0f770.svs', 'TCGA-AA-3855-01Z-00-DX1.f305ce6c-e87c-4e68-b31e-2e6e8b52606f.svs', 'TCGA-AA-3975-01Z-00-DX1.e73492da-b6fb-4c56-ab30-53b0b7041e04.svs', 'TCGA-AD-6548-01Z-00-DX1.4e047481-8926-48e6-9eba-46597c4cc396.svs', 'TCGA-DM-A0XF-01Z-00-DX1.6FD3D3CF-A1E2-4F4E-BF02-F81B1A1061CC.svs', 'TCGA-AA-3678-01Z-00-DX1.4bc052e9-e5b0-4328-afe5-8d19fd2c386c.svs', 'TCGA-F4-6704-01Z-00-DX1.f7cb6641-a6f3-40b7-a5cb-aaf604f73d0f.svs', 'TCGA-G4-6320-01Z-00-DX1.09f11d38-4d47-44c9-b8d6-4d4910c6280e.svs', 'TCGA-D5-5538-01Z-00-DX1.7187bda7-9217-4395-a5e9-647357bc1c3a.svs', 'TCGA-A6-5660-01Z-00-DX1.b254e383-a889-4b73-8f91-8580c8285754.svs', 'TCGA-A6-6651-01Z-00-DX1.09ad2d69-d71d-4fa2-9504-80557a053db4.svs', 'TCGA-F4-6854-01Z-00-DX1.a4d18769-1632-41e4-b49d-4a88d36e21ab.svs', 'TCGA-AA-3976-01Z-00-DX1.d2519da8-bc55-4dde-9839-9fa51ecff1b3.svs', 'TCGA-G4-6627-01Z-00-DX1.f23c6977-d0cb-4bc8-b373-4b1b73c331cc.svs', 'TCGA-AA-3695-01Z-00-DX1.be93a101-7b57-4140-bd82-86c65e65ca27.svs', 'TCGA-NH-A8F8-01Z-00-DX1.0C13D583-0BCE-44F7-A4E6-5994FE97B99C.svs', 'TCGA-AA-3819-01Z-00-DX1.cd674efa-b953-4721-9468-ec6ad8b6f567.svs', 'TCGA-AZ-4313-01Z-00-DX1.5e7ecf69-d1fd-4997-9dcc-ab8e9f10b423.svs', 'TCGA-AA-3850-01Z-00-DX1.49b55930-74fd-4103-9151-7b906a18be02.svs', 'TCGA-AA-3552-01Z-00-DX1.84133d42-9a39-44b5-a1ec-a5382650c939.svs', 'TCGA-CA-6717-01Z-00-DX1.08da75b7-a08f-46b3-a8c0-24f601ec4558.svs', 'TCGA-5M-AAT4-01Z-00-DX1.725C46CA-9354-43AC-AA81-3E5A66354D6B.svs', 'TCGA-AA-A00E-01Z-00-DX1.ABFCAF2D-287A-445F-9F32-BD00D1B385C3.svs', 'TCGA-AY-5543-01Z-00-DX1.f3614d19-8391-49cc-a0e4-932e717696d3.svs', 'TCGA-A6-2679-01Z-00-DX1.8df66ef4-d9e5-41db-836d-f0afe46d6b5a.svs', 'TCGA-AA-A01F-01Z-00-DX1.A09E4A5B-1DD2-472C-B387-91803FEE514A.svs', 'TCGA-AA-3837-01Z-00-DX1.5692e5c0-6dc2-45be-a5f5-00a907c5c824.svs', 'TCGA-D5-5539-01Z-00-DX1.9c46fe78-2adb-4f49-9141-cda135c2c90b.svs', 'TCGA-AA-A01P-01Z-00-DX1.D7AAA4F0-C956-4346-8948-DADACDFB3B69.svs', 'TCGA-AA-3851-01Z-00-DX1.cefbb22e-6b16-41b2-b732-452bf2efe425.svs', 'TCGA-CA-5256-01Z-00-DX1.67cc2ca1-40df-4e76-be88-dfd93e20e017.svs', 'TCGA-DM-A288-01Z-00-DX1.716efd68-52d4-4049-b9a6-480700579e74.svs', 'TCGA-D5-6898-01Z-00-DX1.d2c5ab3b-da4d-4fa4-b535-4339d59515e7.svs', 'TCGA-A6-6140-01Z-00-DX1.f34d99be-25dd-4811-9155-0dbb53e849ac.svs', 'TCGA-G4-6310-01Z-00-DX1.b88472d4-3adc-4e4d-b0f2-0dc195a3d7df.svs', 'TCGA-AA-3562-01Z-00-DX1.e07893e6-646d-41b5-be51-9c19d51f6743.svs', 'TCGA-D5-6932-01Z-00-DX1.d18111de-f0f5-4637-8534-a2b4396cbb41.svs', 'TCGA-F4-6703-01Z-00-DX1.28225f5d-d880-4605-831f-f22ec0272cde.svs', 'TCGA-AA-3495-01Z-00-DX1.67DEE36B-724E-4B4F-B3A9-B4E8CCCEFA80.svs', 'TCGA-AA-3530-01Z-00-DX1.298325a7-f5f8-4b5e-b3da-dd9dab6e4820.svs', 'TCGA-AA-3833-01Z-00-DX1.d27bd30c-bba2-4621-8157-feb28ba2e241.svs', 'TCGA-AA-3872-01Z-00-DX1.eb3732ee-40e3-4ff0-a42b-d6a85cfbab6a.svs', 'TCGA-G4-6303-01Z-00-DX1.5819f041-d9b8-4fd2-bf52-65677be31df1.svs', 'TCGA-AA-3506-01Z-00-DX1.08CFD143-1A4C-4262-831E-79D9C4BBF453.svs', 'TCGA-AA-3821-01Z-00-DX1.019b3b1d-5c05-4be6-af25-6ee63475897e.svs', 'TCGA-G4-6323-01Z-00-DX1.f97c27a6-9fbe-4a81-b4aa-020c25279449.svs', 'TCGA-AD-6888-01Z-00-DX1.47AE342C-4577-4D8B-9048-0B106C5960E7.svs', 'TCGA-CM-5349-01Z-00-DX1.d893eb9a-0321-4052-acfc-8c9a6e463921.svs', 'TCGA-DM-A1D0-01Z-00-DX1.1EE92F9A-3DAA-4C1E-9A17-F9E0D31BE0C1.svs', 'TCGA-F4-6855-01Z-00-DX1.41ed5985-be19-4dce-aab6-de3be0f1dcca.svs', 'TCGA-G4-6299-01Z-00-DX1.22701e3b-7bfb-45ad-9382-842b2da0387a.svs', 'TCGA-A6-2686-01Z-00-DX1.0540a027-2a0c-46c7-9af0-7b8672631de7.svs', 'TCGA-G4-6293-01Z-00-DX1.62ed5ed9-a79a-487a-bd6f-1f3f0571d44d.svs', 'TCGA-NH-A6GC-06Z-00-DX1.5F90CBDB-794B-498D-B75F-60EBDE17B22A.svs', 'TCGA-AA-A010-01Z-00-DX1.AA174B84-1EF2-47EA-85A6-516B3328325D.svs', 'TCGA-A6-2675-01Z-00-DX1.d37847d6-c17f-44b9-b90a-84cd1946c8ab.svs', 'TCGA-AA-A00D-01Z-00-DX1.A4358CDC-9B7E-4802-BF1C-741F533BBD96.svs', 'TCGA-AA-A00U-01Z-00-DX1.E83A6B38-D472-482F-89D7-FF61FB589371.svs', 'TCGA-AA-A01V-01Z-00-DX1.1AC200C2-F577-421D-B91F-F0A7251C3D90.svs', 'TCGA-AA-3519-01Z-00-DX1.82e03504-31d8-43d5-8d3f-01d9016af0fe.svs']


def init_task():
    train_transform, test_transform = init_training_transforms()

    logger, callbacks = init_training_callbacks()

    Logger.log("Loading Datasets..", log_importance=1)
    # loading tile filepaths
    df_tiles = pd.read_csv(Configs.VC_DF_TILE_PATHS_PATH)
    if Configs.VC_COHORT == 'COAD':
        df_tiles = df_tiles[df_tiles.filename.isin(coad_filenames)]
        df_tiles['cohort'] = 'COAD'
    else:
        df_tiles = df_tiles[df_tiles.cohort == Configs.VC_COHORT]
    # loading labels
    df_labels = pd.read_csv(Configs.VC_LABEL_DF_PATH)
    df_labels.rename(columns={'GT_array': 'y'}, inplace=True)
    df_labels.y = df_labels.y.apply(lambda a: torch.Tensor(eval(a)).long())

    df_labels_merged_tiles = df_labels.merge(df_tiles, how='inner', on='patient_id')
    df_labels_merged_tiles['slide_id'] = df_labels_merged_tiles.slide_uuid

    num_snps = len(df_labels.y.iloc[0])
    if Configs.VC_SAMPLE_SNPS is not None:
        model = init_model(Configs.VC_SAMPLE_SNPS)
    else:
        model = init_model(num_snps)

    return df_labels_merged_tiles, train_transform, test_transform, logger, callbacks, model


def init_model(num_snps):
    if Configs.VC_TEST_ONLY is None:
        model = VariantClassifier(output_shape=(3, num_snps), tile_encoder_name=Configs.VC_TILE_ENCODER, class_to_ind=Configs.VC_CLASS_TO_IND,
                                  learning_rate=Configs.VC_INIT_LR, frozen_backbone=Configs.VC_FROZEN_BACKBONE,
                                  num_iters_warmup_wo_backbone=Configs.VC_ITER_TRAINING_WARMUP_WO_BACKBONE)
    else:
        model = VariantClassifier.load_from_checkpoint(Configs.VC_TEST_ONLY, output_shape=(3, num_snps),
                                                       tile_encoder_name=Configs.VC_TILE_ENCODER,
                                                       class_to_ind=Configs.VC_CLASS_TO_IND,
                                                       learning_rate=Configs.VC_INIT_LR,
                                                       frozen_backbone=Configs.VC_FROZEN_BACKBONE,
                                                       num_iters_warmup_wo_backbone=Configs.VC_ITER_TRAINING_WARMUP_WO_BACKBONE)
    return model




