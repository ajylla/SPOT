
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from matplotlib.offsetbox import AnchoredText
from pandas.tseries.frequencies import to_offset
from psp_isois_loader import calc_av_en_flux_PSP_EPIHI, calc_av_en_flux_PSP_EPILO, psp_isois_load
from soho_loader import calc_av_en_flux_ERNE, soho_load
from solo_epd_loader import epd_load
from stereo_loader import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from stereo_loader import calc_av_en_flux_SEPT, stereo_load
from wind_3dp_loader import wind3dp_load


class Event:

    def __init__(self, start_date, end_date, spacecraft, sensor,
                 species, data_level, data_path, threshold=None):

        self.start_date = start_date
        self.end_date = end_date
        self.spacecraft = spacecraft.lower()
        self.sensor = sensor.lower()
        self.species = species.lower()
        self.data_level = data_level.lower()
        self.data_path = data_path + os.sep
        self.threshold = threshold

        # placeholding class attributes for onset_analysis()
        self.flux_series = None
        self.onset_stats = None
        self.onset_found = None
        self.onset = None
        self.peak_flux = None
        self.peak_time = None
        self.fig = None
        self.bg_mean = None
        self.output = {"flux_series": self.flux_series,
                       "onset_stats": self.onset_stats,
                       "onset_found": self.onset_found,
                       "onset": self.onset,
                       "peak_flux": self.peak_flux,
                       "peak_time": self.peak_time,
                       "fig": self.fig,
                       "bg_mean": self.bg_mean
                       }

        self.load_all_viewing()

    def update_onset_attributes(self, flux_series, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean):
        """
        Method to update onset-related attributes, that are None by default and only have values after analyse() has been run.
        """
        self.flux_series = flux_series
        self.onset_stats = onset_stats
        self.onset_found = onset_found
        self.onset = onset_stats[-1]
        self.peak_flux = peak_flux
        self.peak_time = peak_time
        self.fig = fig
        self.bg_mean = bg_mean

        # also remember to update the dictionary, it won't update automatically
        self.output = {"flux_series": self.flux_series,
                       "onset_stats": self.onset_stats,
                       "onset_found": self.onset_found,
                       "onset": self.onset,
                       "peak_flux": self.peak_flux,
                       "peak_time": self.peak_time,
                       "fig": self.fig,
                       "bg_mean": self.bg_mean
                       }

    def load_data(self, spacecraft, sensor, viewing, data_level,
                  autodownload=True, threshold=None):

        if(self.spacecraft == 'solo'):
            df_i, df_e, energs = epd_load(sensor=sensor,
                                          viewing=viewing,
                                          level=data_level,
                                          startdate=self.start_date,
                                          enddate=self.end_date,
                                          path=self.data_path,
                                          autodownload=autodownload)

            return df_i, df_e, energs

        if(self.spacecraft[:2].lower() == 'st'):
            if(self.sensor == 'sept'):
                if self.species in ["p", "i"]:
                    df_i, channels_dict_df_i = stereo_load(instrument=self.sensor,
                                                           startdate=self.start_date,
                                                           enddate=self.end_date,
                                                           spacecraft=self.spacecraft,
                                                           # sept_species=self.species,
                                                           sept_species='p',
                                                           sept_viewing=viewing,
                                                           resample=None,
                                                           path=self.data_path)
                    df_e, channels_dict_df_e = [], []
                    return df_i, df_e, channels_dict_df_i, channels_dict_df_e

                if self.species == "e":
                    df_e, channels_dict_df_e = stereo_load(instrument=self.sensor,
                                                           startdate=self.start_date,
                                                           enddate=self.end_date,
                                                           spacecraft=self.spacecraft,
                                                           # sept_species=self.species,
                                                           sept_species='e',
                                                           sept_viewing=viewing,
                                                           resample=None,
                                                           path=self.data_path)

                    df_i, channels_dict_df_i = [], []
                    return df_i, df_e, channels_dict_df_i, channels_dict_df_e

            if(self.sensor == 'het'):
                df, meta = stereo_load(instrument=self.sensor,
                                       startdate=self.start_date,
                                       enddate=self.end_date,
                                       spacecraft=self.spacecraft,
                                       resample=None,
                                       pos_timestamp='center',
                                       path=self.data_path)
                return df, meta

        if(self.spacecraft.lower() == 'soho'):
            if(self.sensor == 'erne'):
                df, meta = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN",
                                     startdate=self.start_date,
                                     enddate=self.end_date,
                                     path=self.data_path,
                                     resample=None,
                                     pos_timestamp='center')
                return df, meta

        if(self.spacecraft.lower() == 'wind'):
            if(self.sensor == '3dp'):
                df_i, meta_i = wind3dp_load(dataset="WI_SOPD_3DP",
                                            startdate=self.start_date,
                                            enddate=self.end_date,
                                            resample=None,
                                            multi_index=False,
                                            path=self.data_path,
                                            threshold=self.threshold)

                df_e, meta_e = wind3dp_load(dataset="WI_SFPD_3DP",
                                            startdate=self.start_date,
                                            enddate=self.end_date,
                                            resample=None,
                                            multi_index=False,
                                            path=self.data_path,
                                            threshold=self.threshold)

                return df_i, df_e, meta_i, meta_e

        if(self.spacecraft.lower() == 'psp'):
            if(self.sensor.lower() == 'isois-epihi'):
                df, meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60',
                                          startdate=self.start_date,
                                          enddate=self.end_date,
                                          path=self.data_path,
                                          resample=None)
                return df, meta
            if(self.sensor.lower() == 'isois-epilo'):
                df, meta = psp_isois_load(dataset='PSP_ISOIS-EPILO_L2-PE',
                                          startdate=self.start_date,
                                          enddate=self.end_date,
                                          path=self.data_path,
                                          resample=None,
                                          epilo_channel='F',
                                          epilo_threshold=self.threshold)
                return df, meta

    def load_all_viewing(self):

        if(self.spacecraft == 'solo'):

            if(self.sensor in ['het', 'ept']):

                self.df_i_sun, self.df_e_sun, self.energies_sun =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'sun', self.data_level)

                self.df_i_asun, self.df_e_asun, self.energies_asun =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'asun', self.data_level)

                self.df_i_north, self.df_e_north, self.energies_north =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'north', self.data_level)

                self.df_i_south, self.df_e_south, self.energies_south =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'south', self.data_level)

            elif(self.sensor == 'step'):

                self.df_step, self.energies_step =\
                    self.load_data(self.spacecraft, self.sensor, 'None',
                                   self.data_level)

        if(self.spacecraft[:2].lower() == 'st'):

            if(self.sensor == 'sept'):

                self.df_i_sun, self.df_e_sun, self.energies_i_sun, self.energies_e_sun =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'sun', self.data_level)

                self.df_i_asun, self.df_e_asun, self.energies_i_asun, self.energies_e_asun =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'asun', self.data_level)

                self.df_i_north, self.df_e_north, self.energies_i_north, self.energies_e_north =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'north', self.data_level)

                self.df_i_south, self.df_e_south, self.energies_i_south, self.energies_e_south =\
                    self.load_data(self.spacecraft, self.sensor,
                                   'south', self.data_level)

            elif(self.sensor == 'het'):

                self.df_het, self.meta_het =\
                    self.load_data(self.spacecraft, self.sensor, 'None',
                                   self.data_level)
                self.current_df_i = self.df_het.filter(like='Proton')
                self.current_df_e = self.df_het.filter(like='Electron')
                self.current_energies = self.meta_het

        if(self.spacecraft.lower() == 'soho'):

            if(self.sensor.lower() == 'erne'):

                self.df, self.meta =\
                    self.load_data(self.spacecraft, self.sensor, 'None',
                                   self.data_level)
                self.current_df_i = self.df.filter(like='PH_')
                # self.current_df_e = self.df.filter(like='Electron')
                self.current_energies = self.meta

        if(self.spacecraft.lower() == 'wind'):
            if(self.sensor.lower() == '3dp'):
                self.df_i, self.df_e, self.meta_i, self.meta_e = \
                    self.load_data(self.spacecraft, self.sensor, 'None', self.data_level, threshold=self.threshold)
                # self.df_i = self.df_i.filter(like='FLUX')
                # self.df_e = self.df_e.filter(like='FLUX')
                self.current_i_energies = self.meta_i
                self.current_e_energies = self.meta_e

        if(self.spacecraft.lower() == 'psp'):
            if(self.sensor.lower() == 'isois-epihi'):
                # Note: load_data(viewing='all') doesn't really has an effect, but for PSP/ISOIS-EPIHI all viewings are always loaded anyhow.
                self.df, self.meta = self.load_data(self.spacecraft, self.sensor, 'all', self.data_level)
                self.df_e = self.df.filter(like='Electrons_Rate_')
                self.current_e_energies = self.meta
                self.df_i = self.df.filter(like='H_Flux_')
                self.current_i_energies = self.meta
            if(self.sensor.lower() == 'isois-epilo'):
                # Note: load_data(viewing='all') doesn't really has an effect, but for PSP/ISOIS-EPILO all viewings are always loaded anyhow.
                self.df, self.meta = self.load_data(self.spacecraft, self.sensor, 'all', self.data_level, threshold=self.threshold)
                self.df_e = self.df.filter(like='Electron_CountRate_')
                self.current_e_energies = self.meta
                # protons not yet included in PSP/ISOIS-EPILO dataset
                # self.df_i = self.df.filter(like='H_Flux_')
                # self.current_i_energies = self.meta

    def choose_data(self, viewing):

        if(self.spacecraft == 'solo'):
            if(viewing == 'sun'):

                self.current_df_i = self.df_i_sun
                self.current_df_e = self.df_e_sun
                self.current_energies = self.energies_sun

            elif(viewing == 'asun'):

                self.current_df_i = self.df_i_asun
                self.current_df_e = self.df_e_asun
                self.current_energies = self.energies_asun

            elif(viewing == 'north'):

                self.current_df_i = self.df_i_north
                self.current_df_e = self.df_e_north
                self.current_energies = self.energies_north

            elif(viewing == 'south'):

                self.current_df_i = self.df_i_south
                self.current_df_e = self.df_e_south
                self.current_energies = self.energies_south

        if(self.spacecraft[:2].lower() == 'st'):
            if(self.sensor == 'sept'):
                if(viewing == 'sun'):

                    self.current_df_i = self.df_i_sun
                    self.current_df_e = self.df_e_sun
                    self.current_i_energies = self.energies_i_sun
                    self.current_e_energies = self.energies_e_sun

                elif(viewing == 'asun'):

                    self.current_df_i = self.df_i_asun
                    self.current_df_e = self.df_e_asun
                    self.current_i_energies = self.energies_i_asun
                    self.current_e_energies = self.energies_e_asun

                elif(viewing == 'north'):

                    self.current_df_i = self.df_i_north
                    self.current_df_e = self.df_e_north
                    self.current_i_energies = self.energies_i_north
                    self.current_e_energies = self.energies_e_north

                elif(viewing == 'south'):

                    self.current_df_i = self.df_i_south
                    self.current_df_e = self.df_e_south
                    self.current_i_energies = self.energies_i_south
                    self.current_e_energies = self.energies_e_south

        if(self.spacecraft.lower() == 'wind'):
            if(self.sensor.lower() == '3dp'):
                col_list_i = [col for col in self.df_i.columns if col.endswith(str(viewing))]
                col_list_e = [col for col in self.df_e.columns if col.endswith(str(viewing))]
                self.current_df_i = self.df_i[col_list_i]
                self.current_df_e = self.df_e[col_list_e]

        if(self.spacecraft.lower() == 'psp'):
            if(self.sensor.lower() == 'isois-epihi'):
                # viewing = 'A' or 'B'
                self.current_df_e = self.df_e[self.df_e.columns[self.df_e.columns.str.startswith(viewing.upper())]]
                self.current_df_i = self.df_i[self.df_i.columns[self.df_i.columns.str.startswith(viewing.upper())]]
            if(self.sensor.lower() == 'isois-epilo'):
                # viewing = '0' to '7'
                self.current_df_e = self.df_e[self.df_e.columns[self.df_e.columns.str.endswith(viewing)]]
                # protons not yet included in PSP/ISOIS-EPILO dataset
                # self.current_df_i = self.df_i[self.df_i.columns[self.df_i.columns.str.endswith(viewing)]]

    def calc_av_en_flux_HET(self, df, energies, en_channel):

        """This function averages the flux of several
        energy channels of SolO/HET into a combined energy channel
        channel numbers counted from 0

        Parameters
        ----------
        df : pd.DataFrame DataFrame containing HET data
            DataFrame containing HET data
        energies : dict
            Energy dict returned from epd_loader (from Jan)
        en_channel : int or list
            energy channel or list with first and last channel to be used
        species : string
            'e', 'electrons', 'p', 'i', 'protons', 'ions'

        Returns
        -------
        pd.DataFrame
            flux_out: contains channel-averaged flux

        Raises
        ------
        Exception
            [description]
        """

        species = self.species

        try:

            if species not in ['e', 'electrons', 'p', 'protons', 'H']:

                raise ValueError("species not defined. Must by one of 'e',\
                                 'electrons', 'p', 'protons', 'H'")

        except ValueError as error:

            print(repr(error))
            raise

        if species in ['e', 'electrons']:

            en_str = energies['Electron_Bins_Text']
            bins_width = 'Electron_Bins_Width'
            flux_key = 'Electron_Flux'

        if species in ['p', 'protons', 'H']:

            en_str = energies['H_Bins_Text']
            bins_width = 'H_Bins_Width'
            flux_key = 'H_Flux'

            if flux_key not in df.keys():

                flux_key = 'H_Flux'

        if type(en_channel) == list:

            en_channel_string = en_str[en_channel[0]][0].split()[0] + ' - '\
                + en_str[en_channel[-1]][0].split()[2] + ' ' +\
                en_str[en_channel[-1]][0].split()[3]

            if len(en_channel) > 2:

                raise Exception('en_channel must have len 2 or less!')

            if len(en_channel) == 2:

                DE = energies[bins_width]

                for bins in np.arange(en_channel[0], en_channel[-1] + 1):

                    if bins == en_channel[0]:

                        I_all = df[flux_key].values[:, bins] * DE[bins]

                    else:

                        I_all = I_all + df[flux_key].values[:, bins] * DE[bins]

                DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1] + 1)])
                flux_av_en = pd.Series(I_all/DE_total, index=df.index)
                flux_out = pd.DataFrame({'flux': flux_av_en}, index=df.index)

            else:

                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux':
                                        df[flux_key].values[:, en_channel]},
                                        index=df.index)

        else:

            flux_out = pd.DataFrame({'flux':
                                    df[flux_key].values[:, en_channel]},
                                    index=df.index)
            en_channel_string = en_str[en_channel]

        return flux_out, en_channel_string

    def calc_av_en_flux_EPT(self, df, energies, en_channel):

        """This function averages the flux of several energy
        channels of EPT into a combined energy channel
        channel numbers counted from 0

        Parameters
        ----------
        df : pd.DataFrame DataFrame containing EPT data
            DataFrame containing EPT data
        energies : dict
            Energy dict returned from epd_loader (from Jan)
        en_channel : int or list
            energy channel number(s) to be used
        species : string
            'e', 'electrons', 'p', 'i', 'protons', 'ions'

        Returns
        -------
        pd.DataFrame
            flux_out: contains channel-averaged flux

        Raises
        ------
        Exception
            [description]
        """

        species = self.species

        try:

            if species not in ['e', 'electrons', 'p', 'i', 'protons', 'ions']:

                raise ValueError("species not defined. Must by one of 'e',"
                                 "'electrons', 'p', 'i', 'protons', 'ions'")

        except ValueError as error:
            print(repr(error))
            raise

        if species in ['e', 'electrons']:

            bins_width = 'Electron_Bins_Width'
            flux_key = 'Electron_Flux'
            en_str = energies['Electron_Bins_Text']

        if species in ['p', 'i', 'protons', 'ions']:

            bins_width = 'Ion_Bins_Width'
            flux_key = 'Ion_Flux'
            en_str = energies['Ion_Bins_Text']

            if flux_key not in df.keys():

                flux_key = 'H_Flux'

        if type(en_channel) == list:

            en_channel_string = en_str[en_channel[0]][0].split()[0] + ' - '\
                + en_str[en_channel[-1]][0].split()[2] + ' '\
                + en_str[en_channel[-1]][0].split()[3]

            if len(en_channel) > 2:

                raise Exception('en_channel must have len 2 or less!')

            if len(en_channel) == 2:

                DE = energies[bins_width]

                for bins in np.arange(en_channel[0], en_channel[-1]+1):

                    if bins == en_channel[0]:

                        I_all = df[flux_key].values[:, bins] * DE[bins]

                    else:

                        I_all = I_all + df[flux_key].values[:, bins] * DE[bins]

                DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
                flux_av_en = pd.Series(I_all/DE_total, index=df.index)
                flux_out = pd.DataFrame({'flux': flux_av_en}, index=df.index)

            else:

                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux':
                                        df[flux_key].values[:, en_channel]},
                                        index=df.index)

        else:

            flux_out = pd.DataFrame({'flux':
                                    df[flux_key].values[:, en_channel]},
                                    index=df.index)
            en_channel_string = en_str[en_channel]

        return flux_out, en_channel_string

    def resample(self, df_flux, resample_period):

        df_flux_out = df_flux.resample(resample_period, label='left').mean()
        df_flux_out.index = df_flux_out.index\
            + to_offset(pd.Timedelta(resample_period)/2)

        return df_flux_out

    def print_info(self, title, info):

        title_string = "##### >" + title + "< #####"
        print(title_string)
        print(info)
        print('#'*len(title_string) + '\n')

    def mean_value(self, tb_start, tb_end, flux_series):

        """
        This function calculates the classical mean of the background period
        which is used in the onset analysis.
        """

        # replace date_series with the resampled version
        date = flux_series.index
        background = flux_series.loc[(date >= tb_start) & (date < tb_end)]
        mean_value = np.nanmean(background)
        sigma = np.nanstd(background)

        return [mean_value, sigma]

    def onset_determination(self, ma_sigma, flux_series, cusum_window, bg_end_time):

        flux_series = flux_series[bg_end_time:]

        # assert date and the starting index of the averaging process
        date = flux_series.index
        ma = ma_sigma[0]
        sigma = ma_sigma[1]
        md = ma + self.x_sigma*sigma

        # k may get really big if sigma is large in comparison to mean
        try:

            k = (md-ma)/(np.log(md)-np.log(ma))
            k_round = round(k/sigma)

        except ValueError:

            # First ValueError I encountered was due to ma=md=2.0 -> k = "0/0"
            k_round = 1

        # choose h, the variable dictating the "hastiness" of onset alert
        if k < 1.0:

            h = 1

        else:

            h = 2

        alert = 0
        cusum = np.zeros(len(flux_series))
        norm_channel = np.zeros(len(flux_series))

        # set the onset as default to be NaT (Not a Date)
        onset_time = pd.NaT

        for i in range(1, len(cusum)):

            # normalize the observed flux
            norm_channel[i] = (flux_series[i]-ma)/sigma

            # calculate the value for ith cusum entry
            cusum[i] = max(0, norm_channel[i] - k_round + cusum[i-1])

            # check if cusum[i] is above threshold h,
            # if it is -> increment alert
            if cusum[i] > h:

                alert = alert + 1

            else:

                alert = 0

            # cusum_window(default:30) subsequent increments to alert
            # means that the onset was found
            if alert == cusum_window:

                onset_time = date[i - alert]
                break

        # ma = mu_a = background average
        # md = mu_d = background average + 2*sigma
        # k_round = integer value of k, that is the reference value to
        # poisson cumulative sum
        # h = 1 or 2,describes the hastiness of onset alert
        # onset_time = the time of the onset
        # S = the cusum function

        return [ma, md, k_round, norm_channel, cusum, onset_time]

    def onset_analysis(self, df_flux, windowstart, windowlen, channels_dict,
                       channel='flux', cusum_window=30, yscale='log',
                       ylim=None, xlim=None):

        self.print_info("Energy channels", channels_dict)
        spacecraft = self.spacecraft.upper()
        sensor = self.sensor.upper()

        color_dict = {
            'onset_time': '#e41a1c',
            'bg_mean':    '#e41a1c',
            'flux_peak':  '#1a1682',
            'bg':         '#de8585'
        }

        if(self.spacecraft == 'solo'):
            flux_series = df_flux[channel]
        if(self.spacecraft[:2].lower() == 'st'):
            flux_series = df_flux  # [channel]'
        if(self.spacecraft.lower() == 'soho'):
            flux_series = df_flux  # [channel]
        if(self.spacecraft.lower() == 'wind'):
            flux_series = df_flux  # [channel]
        if(self.spacecraft.lower() == 'psp'):
            flux_series = df_flux[channel]
        date = flux_series.index

        if ylim is None:

            ylim = [np.nanmin(flux_series[flux_series > 0]),
                    np.nanmax(flux_series) * 3]

        # dates for start and end of the averaging processes
        avg_start = date[0] + datetime.timedelta(hours=windowstart)
        # ending time is starting time + a given timedelta in hours
        avg_end = avg_start + datetime.timedelta(hours=windowlen)

        if xlim is None:

            xlim = [date[0], date[-1]]

        else:

            df_flux = df_flux[xlim[0]:xlim[-1]]

        # onset not yet found
        onset_found = False
        background_stats = self.mean_value(avg_start, avg_end, flux_series)
        onset_stats =\
            self.onset_determination(background_stats, flux_series,
                                     cusum_window, avg_end)

        if not isinstance(onset_stats[-1], pd._libs.tslibs.nattype.NaTType):

            onset_found = True

        if(self.spacecraft == 'solo'):
            df_flux_peak = df_flux[df_flux[channel] == df_flux[channel].max()]
        if(self.spacecraft[:2].lower() == 'st'):
            df_flux_peak = df_flux[df_flux == df_flux.max()]
        if(self.spacecraft == 'soho'):
            df_flux_peak = df_flux[df_flux == df_flux.max()]
        if(self.spacecraft == 'wind'):
            df_flux_peak = df_flux[df_flux == df_flux.max()]
        if(self.spacecraft == 'psp'):
            # df_flux_peak = df_flux[df_flux == df_flux.max()]
            df_flux_peak = df_flux[df_flux[channel] == df_flux[channel].max()]
        self.print_info("Flux peak", df_flux_peak)
        self.print_info("Onset time", onset_stats[-1])
        self.print_info("Mean of background intensity",
                        background_stats[0])
        self.print_info("Std of background intensity",
                        background_stats[1])

        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['font.size'] = 16
        fig, ax = plt.subplots()
        ax.plot(flux_series.index, flux_series.values, ds='steps-mid')

        # CUSUM and norm datapoints in plots.
        '''
        ax.scatter(flux_series.index, onset_stats[-3], s=1,
                   color='darkgreen', alpha=0.7, label='norm')
        ax.scatter(flux_series.index, onset_stats[-2], s=3,
                   c='maroon', label='CUSUM')
        '''

        # onset time
        if onset_found:

            # Onset time line
            ax.axvline(onset_stats[-1], linewidth=1.5,
                       color=color_dict['onset_time'], linestyle='-',
                       label="Onset time")

        # Flux peak line (first peak only, if there's multiple)
        ax.axvline(df_flux_peak.index[0], linewidth=1.5,
                   color=color_dict['flux_peak'], linestyle='-',
                   label="Peak time")

        # background mean
        ax.axhline(onset_stats[0], linewidth=2,
                   color=color_dict['bg_mean'], linestyle='--',
                   label="Mean of background")

        # background mean + 2*std
        ax.axhline(onset_stats[1], linewidth=2,
                   color=color_dict['bg_mean'], linestyle=':',
                   label=f"Mean + {str(self.x_sigma)} * std of background")

        # Background shaded area
        ax.axvspan(avg_start, avg_end, color=color_dict['bg'],
                   label="Background")

        ax.set_xlabel("Time (HH:MM \nYYYY-mm-dd)", fontsize=16)
        ax.set_ylabel(r"Intensity [1/(cm$^{2}$ sr s MeV)]", fontsize=16)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        # figure limits and scale
        plt.ylim(ylim)
        plt.xlim(xlim[0], xlim[1])
        plt.yscale(yscale)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                   fancybox=True, shadow=False, ncol=3, fontsize=16)

        # tickmarks, their size etc...
        plt.tick_params(which='major', length=5, width=1.5, labelsize=16)
        plt.tick_params(which='minor', length=4, width=1)

        # date tick locator and formatter
        ax.xaxis_date()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(9))
        utc_dt_format1 = DateFormatter('%H:%M \n%Y-%m-%d')
        ax.xaxis.set_major_formatter(utc_dt_format1)

        if self.species == 'e':

            s_identifier = 'electrons'

        if self.species in ['p', 'i']:

            if((spacecraft == 'sta' and sensor == 'sept') or (spacecraft == 'solo' and sensor == 'ept')):

                s_identifier = 'ions'

            else:

                s_identifier = 'protons'

        self.print_info("Particle species", s_identifier)

        if(self.viewing_used != '' and self.viewing_used != None):

            plt.title(f"{spacecraft}/{sensor} {channels_dict} {s_identifier}\n"
                      f"{self.averaging_used} averaging, viewing: "
                      f"{self.viewing_used.upper()}")

        else:

            plt.title(f"{spacecraft}/{sensor} {channels_dict} {s_identifier}\n"
                      f"{self.averaging_used} averaging")

        fig.set_size_inches(16, 8)

        # Onset label
        if(onset_found):

            if(self.spacecraft == 'solo' or self.spacecraft == 'psp'):
                plabel = AnchoredText(f"Onset time: {str(onset_stats[-1])[:19]}\n"
                                      f"Peak flux: {df_flux_peak['flux'][0]:.2E}",
                                      prop=dict(size=13), frameon=True,
                                      loc=(4))
            # if(self.spacecraft[:2].lower() == 'st' or self.spacecraft == 'soho' or self.spacecraft == 'wind'):
            else:
                plabel = AnchoredText(f"Onset time: {str(onset_stats[-1])[:19]}\n"
                                      f"Peak flux: {df_flux_peak.values[0]:.2E}",
                                      prop=dict(size=13), frameon=True,
                                      loc=(4))

        else:

            plabel = AnchoredText("No onset found",
                                  prop=dict(size=13), frameon=True,
                                  loc=(4))

        plabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        plabel.patch.set_linewidth(2.0)

        # Background label
        blabel = AnchoredText(f"Background:\n{avg_start} - {avg_end}",
                              prop=dict(size=13), frameon=True,
                              loc='upper left')
        blabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        blabel.patch.set_linewidth(2.0)

        # Energy and species label
        '''
        eslabel = AnchoredText(f"{channels_dict} {s_identifier}",
                               prop=dict(size=13), frameon=True,
                               loc='lower left')
        eslabel.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
        eslabel.patch.set_linewidth(2.0)
        '''

        ax.add_artist(plabel)
        ax.add_artist(blabel)
        # ax.add_artist(eslabel)
        plt.tight_layout()
        plt.show()

        return flux_series, onset_stats, onset_found, df_flux_peak, df_flux_peak.index[0], fig, background_stats[0]

    def analyse(self, viewing, bg_start, bg_length, resample_period=None,
                channels=[0, 1], yscale='log', cusum_window=30, xlim=None, x_sigma=2):

        if isinstance(channels, int):
            channels = (channels,)

        if (self.spacecraft[:2].lower() == 'st' and self.sensor == 'sept') \
                or (self.spacecraft.lower() == 'psp' and self.sensor.startswith('isois')) \
                or (self.spacecraft.lower() == 'solo' and self.sensor == 'ept') \
                or (self.spacecraft.lower() == 'solo' and self.sensor == 'het') \
                or (self.spacecraft.lower() == 'wind' and self.sensor == '3dp'):
            self.viewing_used = viewing
            self.choose_data(viewing)
        elif (self.spacecraft[:2].lower() == 'st' and self.sensor == 'het'):
            self.viewing_used = ''
        elif (self.spacecraft.lower() == 'soho' and self.sensor == 'erne'):
            self.viewing_used = ''

        self.averaging_used = resample_period
        self.x_sigma = x_sigma

        if(self.spacecraft == 'solo'):

            if(self.sensor == 'het'):

                if(self.species in ['p', 'i']):

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_HET(self.current_df_i,
                                                 self.current_energies,
                                                 channels)
                elif(self.species == 'e'):

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_HET(self.current_df_e,
                                                 self.current_energies,
                                                 channels)

            elif(self.sensor == 'ept'):

                if(self.species in ['p', 'i']):

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_EPT(self.current_df_i,
                                                 self.current_energies,
                                                 channels)
                elif(self.species == 'e'):

                    df_flux, en_channel_string =\
                        self.calc_av_en_flux_EPT(self.current_df_e,
                                                 self.current_energies,
                                                 channels)

            else:
                invalid_sensor_msg = "Invalid sensor!"
                raise Exception(invalid_sensor_msg)

        if(self.spacecraft[:2] == 'st'):

            if(self.sensor == 'het'):

                if(self.species in ['p', 'i']):

                    df_flux, en_channel_string =\
                        calc_av_en_flux_ST_HET(self.current_df_i,
                                               self.current_energies['channels_dict_df_p'],
                                               channels,
                                               species='p')
                elif(self.species == 'e'):

                    df_flux, en_channel_string =\
                        calc_av_en_flux_ST_HET(self.current_df_e,
                                               self.current_energies['channels_dict_df_e'],
                                               channels,
                                               species='e')

            elif(self.sensor == 'sept'):

                if(self.species in ['p', 'i']):

                    df_flux, en_channel_string =\
                        calc_av_en_flux_SEPT(self.current_df_i,
                                             self.current_i_energies,
                                             channels)
                elif(self.species == 'e'):

                    df_flux, en_channel_string =\
                        calc_av_en_flux_SEPT(self.current_df_e,
                                             self.current_e_energies,
                                             channels)

        if(self.spacecraft == 'soho'):

            if(self.sensor == 'erne'):

                if(self.species in ['p', 'i']):

                    df_flux, en_channel_string =\
                        calc_av_en_flux_ERNE(self.current_df_i,
                                             self.current_energies['channels_dict_df_p'],
                                             channels,
                                             species='p',
                                             sensor='HET')

        if(self.spacecraft == 'wind'):
            if(self.sensor == '3dp'):
                # convert single-element "channels" list to integer
                if type(channels) == list:
                    if len(channels) == 1:
                        channels = channels[0]
                    else:
                        print("No multi-channel support for Wind/3DP included yet! Select only one single channel.")
                if(self.species in ['p', 'i']):
                    df_flux = self.current_df_i.filter(like=f'FLUX_E{channels}')
                    # extract pd.Series for further use:
                    df_flux = df_flux[df_flux.columns[0]]
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    df_flux = df_flux*1e6
                    en_channel_string = self.current_i_energies['channels_dict_df']['Bins_Text'][f'ENERGY_{channels}']
                elif(self.species == 'e'):
                    df_flux = self.current_df_e.filter(like=f'FLUX_E{channels}')
                    # extract pd.Series for further use:
                    df_flux = df_flux[df_flux.columns[0]]
                    # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                    df_flux = df_flux*1e6
                    en_channel_string = self.current_e_energies['channels_dict_df']['Bins_Text'][f'ENERGY_{channels}']

        if(self.spacecraft.lower() == 'psp'):
            if(self.sensor.lower() == 'isois-epihi'):
                if(self.species in ['p', 'i']):
                    # We're using here only the HET instrument of EPIHI (and not LET1 or LET2)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPIHI(df=self.current_df_i,
                                                  energies=self.current_i_energies,
                                                  en_channel=channels,
                                                  species='p',
                                                  instrument='het',
                                                  viewing=viewing.upper())
                if(self.species == 'e'):
                    # We're using here only the HET instrument of EPIHI (and not LET1 or LET2)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPIHI(df=self.current_df_e,
                                                  energies=self.current_e_energies,
                                                  en_channel=channels,
                                                  species='e',
                                                  instrument='het',
                                                  viewing=viewing.upper())
            if(self.sensor.lower() == 'isois-epilo'):
                if(self.species == 'e'):
                    # We're using here only the F channel of EPILO (and not E or G)
                    df_flux, en_channel_string =\
                        calc_av_en_flux_PSP_EPILO(df=self.current_df_e,
                                                  en_dict=self.current_e_energies,
                                                  en_channel=channels,
                                                  species='e',
                                                  mode='pe',
                                                  chan='F',
                                                  viewing=viewing)

        if(resample_period is not None):

            df_averaged = self.resample(df_flux, resample_period)

        else:

            df_averaged = df_flux

        flux_series, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean =\
            self.onset_analysis(df_averaged, bg_start, bg_length,
                                en_channel_string, yscale=yscale, cusum_window=cusum_window, xlim=xlim)

        # update class attributes before returning variables:
        self.update_onset_attributes(flux_series, onset_stats, onset_found, peak_flux.values[0], peak_time, fig, bg_mean)

        return flux_series, onset_stats, onset_found, peak_flux, peak_time, fig, bg_mean
