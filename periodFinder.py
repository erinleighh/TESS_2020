import glob
import os
import pandas as pd
import numpy as np
import exoplanet as xo
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle, BoxLeastSquares
import time as timer

start = timer.time()


def lombscargle(time, relFlux):
    LS = LombScargle(time, relFlux)
    frequency, power = LS.autopower()
    bestPeriod = 1 / frequency[np.argmax(power)]
    maxPower = np.max(power)
    period = 1 / frequency

    return period, power, bestPeriod, maxPower


def autocorrelationfn(time, relFlux, relFluxErr):
    acf = xo.autocorr_estimator(time.values, relFlux.values, yerr=relFluxErr.values, max_peaks=10)

    period = acf['autocorr'][0]
    power = acf['autocorr'][1]

    acfPowerPd = pd.DataFrame(power)
    acfLocalMaxima = acfPowerPd[(acfPowerPd.shift(1) < acfPowerPd) & (acfPowerPd.shift(-1) < acfPowerPd)]
    maxPower = np.max(acfLocalMaxima).values

    bestPeriod = period[np.where(power == maxPower)[0]][0]
    peaks = acf['peaks'][0]['period']

    if len(acf['peaks']) > 0:
        window = int(peaks / np.abs(np.nanmedian(np.diff(time))) / 5)
    else:
        window = 128

    return period, power, bestPeriod, maxPower, window


def boxleastsquares(time, relFlux, relFluxErr, acfBP):
    model = BoxLeastSquares(time.values, relFlux.values, dy=relFluxErr.values)
    # duration = [20 / 1440, 40 / 1440, 80 / 1440]
    periodogram = model.autopower([80 / 1440], method='fast', objective='snr', minimum_n_transit=3)
    period = periodogram.period
    power = periodogram.power
    maxPower = np.max(periodogram.power)
    bestPeriod = periodogram.period[np.argmax(periodogram.power)]

    return period, power, bestPeriod, maxPower


def addtotable(table, oID, lsBP, lsMP, blsBP, blsMP, acfBP, acfMP, f):
    table = table.append(
        {'Obj ID': oID, 'LS Best Per': lsBP, 'LS Max Pow': lsMP, 'ACF Best Per': acfBP, 'ACF Max Pow': acfMP, 'BLS Best Per': blsBP, 'BLS Max Pow': blsMP, 'Filename': f}, ignore_index=True)
    return table


def makegraph(xaxis, yaxis, xlabels, ylabels, lbl, color, marker=None, size=None, style=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if style is None:
        ax.scatter(xaxis, yaxis, color=color, marker=marker, s=size)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    else:
        ax.plot(xaxis, yaxis, color=color)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.title(lbl)
    return ax


data = pd.read_csv('EBfiles.csv')
objects = data['Obj ID'].drop_duplicates()
EBtable = pd.DataFrame(
    columns=['Obj ID', 'LS Best Per', 'LS Max Pow', 'ACF Best Per', 'ACF Max Pow', 'BLS Best Per', 'BLS Max Pow', 'Filename'])

for obj in objects:
    print("\nReading in " + obj)
    objTable = data.loc[data['Obj ID'] == obj]
    files = objTable['Filename'].copy()

    # Store data for all sectors of a given TIC ID
    fullCurve = pd.DataFrame()

    for file in files:
        fitsTable = fits.open(file, memmap=True)
        curveTable = Table(fitsTable[1].data).to_pandas()
        curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
        fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
        curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
        curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

        fullCurve = fullCurve.append(curveData, ignore_index = True)
    
    # Plot full LC
    plt.figure(figsize=(20, 6))
    figName = obj + '.png'
    makegraph(fullCurve['TIME'], fullCurve['REL_FLUX'], 'BJD - 2457000 (days)', 'Relative Flux', 'Light Curve for ' + obj, 'tab:purple', '.', .2)
    plt.savefig(os.path.join('EBs', figName), orientation='landscape')
    plt.close()
    
    # Lomb-Scargle Function
    print("Generating LS periodogram.")
    lsPeriod, lsPower, lsBestPeriod, lsMaxPower = lombscargle(curveData['TIME'], curveData['REL_FLUX'])

    # Autocorrelation Function
    print("Generating ACF periodogram.")
    acfPeriod, acfPower, acfBestPeriod, acfMaxPower, s_window = autocorrelationfn(curveData['TIME'], curveData['REL_FLUX'], curveData['REL_FLUX_ERR'])

    # Box Least Squares
    print("Generating BLS periodogram.")
    BLSperiod, BLSpower, BLSbestPeriod, BLSmaxPower = boxleastsquares(curveData['TIME'], curveData['REL_FLUX'], curveData['REL_FLUX_ERR'], acfBestPeriod)
    
    # Add to table
    print("Adding to table.")
    EBtable = addtotable(EBtable, obj, lsBestPeriod, lsMaxPower, acfBestPeriod, acfMaxPower[0], BLSbestPeriod, BLSmaxPower, file)
        
    print(obj + " complete.")
    
# Print table to file
print("\nPrint curve table to file.\n")
EBtable.to_csv('EBtable.csv')
    

print("\nProcess complete.\n")

end = timer.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))