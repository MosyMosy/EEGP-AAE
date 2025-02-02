from __future__ import print_function
#This is to train the network by running: python single1_train.py <experiment_name> <obj_id>
#e.g.: python single1_train.py subdiv_29_softmax_edge 29

#Prerequisite before training:
#cfg file: <experiment_name>.cfg under the path: path_workspace/cfg/<experiment_name>.cfg
#Rendered foreground imgs to train the AE(Generated by render_training.py) under the path: path_workspace/tmp_datasets/<name_fg_data>.npz
#Random background imgs under the path: path_workspace/tmp_datasets/prepared_bg_imgs.npy
#Rendered imgs under reference rotations R_c(Generated by render_codebook.py) under the path: path_embedding_data (Actually optional, but used in this program for visualization)
import os
import numpy as np
import tensorflow as tf

import configparser
import argparse
import signal
import shutil
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import progressbar

import utils as u
from utils import lazy_property
from dataset import Dataset

from six.moves import range
from network import build_dataset,build_decoder,build_encoder,VectorQuantizerEMA

################################################################################
path_workspace = './ws/'
path_embedding_data = "embeding/training/" #'../../../Edge-Network/embedding20s/'
name_fg_data='prepared_training_data_{:02d}_subdiv'

tau=0.07
lambda_reconst=250.
d_ema=0.99

# Set hyper-parameters.
batch_size = 64
image_size = 128
embedding_dim = 128

learning_rate = 2e-4
num_training_updates = 30000
normalize_images=True #Default false for T-LESS CAD meshes, and True for texture meshes as LINEMOD meshes

if path_workspace == None:
    print('Please define a workspace path:\n')
    exit(-1)

gentle_stop = np.array((1,), dtype=np.bool)
gentle_stop[0] = False

def on_ctrl_c(signal, frame):
    gentle_stop[0] = True


signal.signal(signal.SIGINT, on_ctrl_c)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", default="T-Less", required=False)
parser.add_argument("--obj_id", default="1", required=False)
arguments = parser.parse_args()

full_name = arguments.experiment_name.split('/')
obj_id=arguments.obj_id

experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''


cfg_file_path = u.get_config_file_path(path_workspace, experiment_name, experiment_group)
list_models=[int(obj_id)]

log_dir = u.get_log_dir(path_workspace, experiment_name, experiment_group)
ckpt_dir = os.path.join(log_dir, 'checkpoints_lambda{:d}'.format(int(lambda_reconst)))
checkpoint_file = u.get_checkpoint_basefilename(ckpt_dir)
train_fig_dir = os.path.join(log_dir, 'train_figures_lambda{:d}'.format(int(lambda_reconst)))
dataset_path = u.get_dataset_path(path_workspace)
print('dataset_path',dataset_path)

if not os.path.exists(cfg_file_path):
    print('Could not find config file:\n')
    print('{}\n'.format(cfg_file_path))
    exit(-1)

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(train_fig_dir):
    os.makedirs(train_fig_dir)
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)


args = configparser.ConfigParser()
args.read(cfg_file_path)

shutil.copyfile(cfg_file_path, os.path.join(log_dir,experiment_name+'.cfg'))
tf.reset_default_graph()


# The higher this value, the higher the capacity in the information bottleneck.
dataset = build_dataset(dataset_path,name_fg_data,path_embedding_data+'{:02d}',list_models,args)
dataset.load_training_images()
dataset.load_bg_images()
dataset.load_codebook_rotation(path_codebook_rotation=path_embedding_data+'{:02d}/rot_infos.npz'.format(list_models[0]))
dataset.load_embedding_images()
dataset.compute_knn_rot_embedding_indices(knn=1, use_probability=False)

num_embeddings = dataset.embedding_size
# Build modules.
with tf.variable_scope(experiment_name):#.split('_')[0] + '_' + experiment_name.split('_')[1]):
    ci=dataset.inshape[-1]
    co=dataset.outshape[-1]
    print('ci/co: ',ci, co)

    #################Normalize images###########################
    bgr_y=tf.placeholder(tf.uint8, shape=(image_size, image_size, 3))
    _normalized_bgr_y= tf.reshape(tf.image.per_image_standardization(bgr_y),[image_size,image_size,3])
    min_normalized_bgry=tf.reduce_min(_normalized_bgr_y)
    max_normalized_bgry=tf.reduce_max(_normalized_bgr_y)
    normalized_bgr_y=(_normalized_bgr_y-min_normalized_bgry)/(max_normalized_bgry-min_normalized_bgry)
    ############################################################

    x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, ci))
    pose_label= tf.placeholder(tf.int32, shape=(None, 1))
    y_gt_ae = tf.placeholder(tf.float32, shape=(None, image_size, image_size, co))

    ppixel_edge_w_ae =tf.placeholder(dtype=tf.float32, shape=(None, image_size, image_size, 1))
    gprior_decay = tf.placeholder(tf.float32, shape=None)
    gprior_temperature=tf.placeholder(tf.float32,shape=None)
    gprior_lambda_reconst_cost = tf.placeholder(tf.float32, shape=None)

    print('xshape', x.shape)
    with tf.variable_scope('encoder'):
        encoder = build_encoder(args)

    z = encoder(x)
    decoder_input = z

    pose_embeds = VectorQuantizerEMA(embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    # For training
    with tf.variable_scope('embedding_lookup'):
        pose_train_retr = pose_embeds(z, decay=gprior_decay, temperature=gprior_temperature,encoding_1nn_indices=pose_label, encodings=None, is_training=True)

    total_loss=0.
    with tf.variable_scope('decoder'):
        decoder = build_decoder(args)

    # Decoder for ae
    y_recon_ae = decoder(decoder_input)
    recon_error_ae=0.
    if co in [3, 4]:
        if co == 4:
            reconst_target_bgr = tf.slice(y_gt_ae, [0, 0, 0, 0], [-1, -1, -1, 3])
        else:
            reconst_target_bgr = y_gt_ae

        bootstrap_ratio=4
        y_gt_flat = tf.contrib.layers.flatten(reconst_target_bgr)
        print(reconst_target_bgr.shape)
        print(y_recon_ae['x_bgr'].shape)
        y_reconae_flat = tf.contrib.layers.flatten(y_recon_ae['x_bgr'])
        l2_ae = tf.losses.mean_squared_error(
            y_reconae_flat,
            y_gt_flat,
            reduction=tf.losses.Reduction.NONE
        )
        l2_val_ae, _ = tf.nn.top_k(l2_ae, k=l2_ae.shape[1] // bootstrap_ratio) #py2 to py3
        recon_error_ae_bgr = tf.reduce_mean(l2_val_ae)

        weight_bgr_loss=1.
        recon_error_ae += recon_error_ae_bgr * weight_bgr_loss
        tf.summary.scalar('recons_loss_ae_bgr', recon_error_ae_bgr)

    if co in [1, 4]:
        if co == 4:
            reconst_target_edge = tf.slice(y_gt_ae, [0, 0, 0, 3], [-1, -1, -1, 1])
        else:
            reconst_target_edge = y_gt_ae

        x_flat = tf.contrib.layers.flatten(y_recon_ae['x_edge'])
        reconstruction_label_flat = tf.contrib.layers.flatten(reconst_target_edge)

        loss_weight_flat = tf.contrib.layers.flatten(ppixel_edge_w_ae)
        loss_ce_ori = tf.nn.sigmoid_cross_entropy_with_logits(labels=reconstruction_label_flat, logits=x_flat,name='loss_cross_entropy')

        loss_ce_combined = tf.multiply(loss_weight_flat, loss_ce_ori)
        recon_error_ae_edge = tf.reduce_mean(loss_ce_combined)

        weight_edge_loss=1
        recon_error_ae += recon_error_ae_edge * weight_edge_loss
        tf.summary.scalar('recons_loss_ae_edge', recon_error_ae_edge)

    total_loss += recon_error_ae * 1.
    total_loss+=(pose_train_retr["loss"])/gprior_lambda_reconst_cost
    pose_cosine_loss = tf.reduce_mean((tf.nn.l2_normalize(tf.stop_gradient(pose_train_retr["quantize_1nn"]), dim=1) - tf.nn.l2_normalize(tf.stop_gradient(z), dim=1)) ** 2)
    tf.summary.histogram('pose z', z)

    tf.summary.scalar('recons_loss_ae', recon_error_ae)
    tf.summary.scalar('pose_ce_loss', pose_train_retr["loss"])
    tf.summary.scalar('pose_cosine_err',pose_cosine_loss)
    tf.summary.scalar('temperature',gprior_temperature)
    tf.summary.scalar('lambda_reconst_cost', gprior_lambda_reconst_cost)

    mean, var = tf.nn.moments(pose_embeds.embeddings, 1)
    tf.summary.histogram('pose codebook mean', mean)
    tf.summary.histogram('pose codebook var', var)
    tf.summary.histogram('pose codebook', pose_embeds.embeddings)


    # For evaluation, make sure is_training=False!
    with tf.variable_scope('validation'):
        pose_train_retr_eval = pose_embeds(z,encoding_1nn_indices=None, encodings=None, is_training=False)
        decoder_input_center = pose_train_retr_eval['quantize_1nn']
        x_recon_eval_ae = decoder(decoder_input)

    # Create optimizer and TF session.
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op_vq = tf.contrib.training.create_train_op(total_loss, optim, global_step=global_step)
    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=50)
    merged_summary = tf.summary.merge_all()

# Train.
train_res_recon_error = []
train_res_perplexity = []

widgets = ['Training: ', progressbar.Percentage(),
           ' ', progressbar.Bar(),
           ' ', progressbar.Counter(), ' / %s' % num_training_updates,
           ' ', progressbar.ETA(), ' ']
bar = progressbar.ProgressBar(maxval=num_training_updates, widgets=widgets)

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)

use_clean = False


with tf.Session(config=config) as sess:
    if normalize_images:
        widgets2 = ['Normalization: ', progressbar.Percentage(),
                    ' ', progressbar.Bar(),
                    ' ', progressbar.Counter(), ' / %s' % dataset.train_y.shape[0],
                    ' ', progressbar.ETA(), ' ']
        bar2 = progressbar.ProgressBar(maxval=dataset.train_y.shape[0], widgets=widgets2)
        bar2.start()
        # print(dataset.train_y.shape[0],'start normalization')
        for ii in range(0, dataset.train_y.shape[0]):
            bar2.update(ii)
            # tf_normalized_bgry=tf.image.per_image_standardization(dataset.train_y[ii])
            normalized_bgry = sess.run(normalized_bgr_y, feed_dict={bgr_y: dataset.train_y[ii]})
            normalized_bgrx = sess.run(normalized_bgr_y, feed_dict={bgr_y: dataset.train_x[ii]})
            dataset.train_y[ii, :, :, :] = (normalized_bgry.copy() * 255.).astype(np.uint8)
            dataset.train_x[ii, :, :, :] = (normalized_bgrx.copy() * 255.).astype(np.uint8)
        bar2.finish()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)
    bar.start()
    for i in range(num_training_updates):
        bar.update(i)
        this_x, this_y, this_ppixel_weight, this_obj_label, this_pose_label, this_pose_img = dataset.batch(batch_size, batchx_clean=use_clean,stack_codebook=False)
        this_x = this_x.astype(np.float32)
        this_y = this_y.astype(np.float32)

        feed_dict = {x: this_x, pose_label: this_pose_label, y_gt_ae: this_y, ppixel_edge_w_ae:this_ppixel_weight,
                    gprior_decay: d_ema, gprior_lambda_reconst_cost: lambda_reconst,gprior_temperature: tau}

        results = sess.run([merged_summary, train_op_vq, global_step], feed_dict=feed_dict)
        gs = results[-1]
        if i % 10 == 0:
            summary_writer.add_summary(results[0], gs)

        if (i + 1) % 1000 == 0 or i == 0:
            # sess.run([assign_embedding])
            if (i + 1)%10000==0 or i==0:
                saver.save(sess, checkpoint_file, global_step=i)

            valid_originals, valid_y_gtae, valid_y_gtref, = this_x, this_y, this_pose_img
            valid_originals = valid_originals.astype(np.float32)

            valid_reconstructions_ae,valid_reconstructions_eae = \
                    sess.run([x_recon_eval_ae['x_bgr'],x_recon_eval_ae['x_edge_vis']],feed_dict={x:valid_originals})

            def convert_batch_to_image_grid(image_batch):
                if image_batch.ndim == 3:
                    image_batch = np.expand_dims(image_batch[0:32,:,:], -1).reshape((4, 8, image_size, image_size, 1))
                reshaped = (image_batch[0:32,:,:,:].reshape(4, 8, image_size, image_size, -1)
                            .transpose(0, 2, 1, 3, 4)
                            .reshape(4 * image_size, 8 * image_size, -1))
                return reshaped


            cv2.imwrite(os.path.join(train_fig_dir, '{:05d}_validation_ae_ori_bgr.png'.format(i)),
                        (convert_batch_to_image_grid(valid_originals[:, :, :, 0:3]) * 255.).astype(np.uint8))
            cv2.imwrite(os.path.join(train_fig_dir, '{:05d}_validation_ae_rec_bgr.png'.format(i)),
                        (convert_batch_to_image_grid(valid_reconstructions_ae[:, :, :, 0:3]) * 255.).astype(np.uint8))
            cv2.imwrite(os.path.join(train_fig_dir, '{:05d}_validation_ae_rgt_bgr.png'.format(i)),
                        (convert_batch_to_image_grid(valid_y_gtae[:, :, :, 0:3]) * 255.).astype(np.uint8))
            cv2.imwrite(os.path.join(train_fig_dir, '{:05d}_validation_ae_refgt_bgr.png'.format(i)),
                        (convert_batch_to_image_grid(valid_y_gtref[:, :, :, 0:3])).astype(np.uint8))

            cv2.imwrite(os.path.join(train_fig_dir, '{:05d}_validation_ae_ori_edge.png'.format(i)),
                        (convert_batch_to_image_grid(valid_originals[:, :, :, -1]) * 255.).astype(np.uint8))
            cv2.imwrite(os.path.join(train_fig_dir, '{:05d}_validation_ae_rec_edge.png'.format(i)),
                        (convert_batch_to_image_grid(valid_reconstructions_eae) * 255.).astype(np.uint8))
            cv2.imwrite(os.path.join(train_fig_dir, '{:05d}_validation_ae_rgt_edge.png'.format(i)),
                        (convert_batch_to_image_grid(valid_y_gtae[:, :, :, -1]) * 255.).astype(np.uint8))

        if gentle_stop[0]:
            break

    bar.finish()
