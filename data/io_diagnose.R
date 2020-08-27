# Install packages (if needed)
# x_i<-c("rjson", "ggplot2", "reshape", "gridExtra")
# lapply(x_i, require, character.only = TRUE)
# Load packages
x_r<-c("rjson", "ggplot2", "reshape", "gridExtra")
lapply(x_r, require, character.only = TRUE)

# Load output data
dat <- fromJSON(file = "recommendations.json")
# Retrieve input/ouput data
dat_io <- dat$`input output data`
# Retrieve choice frequency data
dat_freq <- dat$`predictions`
# Set the measure names list
dat_meas <- names(dat$`predictions`)

# Initialize subsets of the data for each input or output variable o f interest
# Input: max change in demand during event/precooling period
dmd_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`demand`[[dat_meas[1]]])))
dmd_dat_pc <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`demand precool`[[dat_meas[1]]])))
# Input: total change in economic benefit during event/precooling period
cost_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`cost`[[dat_meas[1]]])))
cost_dat_pc <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`cost precool`[[dat_meas[1]]])))
# Input: max change in temperature during event/precooling period
temp_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`temperature`[[dat_meas[1]]])))
# temp_dat_pc <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`temperature precool (low)`[[dat_meas[1]]])))
# Input: max change in lighting during event
lgt_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`lighting`[[dat_meas[1]]])))
# Output: change in selection probability
chc_dat <- data.frame(matrix(NA, length(dat_meas), length(dat_io$`choice probabilities`[[dat_meas[1]]])))
chc_dat_pts <- matrix(NA, length(dat_meas))
# Fill in initialized data frames with simulated data
for (r in (1:(length(dat_meas)))){
	# Demand (event/precool)
	dmd_dat[r, ] = dat_io$`demand`[[dat_meas[r]]]
	dmd_dat_pc[r, ] = -dat_io$`demand precool`[[dat_meas[r]]]
	# Economic benefit (overall/precool)
	cost_dat[r, ] = dat_io$`cost`[[dat_meas[r]]]
	cost_dat_pc[r, ] = -dat_io$`cost precool`[[dat_meas[r]]]
	# Temperature (event/precool)
	temp_dat[r, ] = dat_io$`temperature`[[dat_meas[r]]]
	# temp_dat_pc[r, ] = dat_io$`temperature precool (low)`[[dat_meas[r]]]
	# Lighting
	lgt_dat[r, ] = dat_io$`lighting`[[dat_meas[r]]]
	# Choice probability
	chc_dat[r, ] = dat_io$`choice probabilities`[[dat_meas[r]]]
	chc_dat_pts[r] = dat_freq[[dat_meas[r]]]
}

# Plot the data
jpeg("IO_Diagnostics.jpeg", width = 25, height=16, units="in", res=200)
# Set measure names to use in grouping each of the variable data
dmd_dat$group <- dmd_dat_pc$group <- cost_dat$group <- cost_dat_pc$group <- 
temp_dat$group <- lgt_dat$group <- chc_dat$group <- dat_meas

# Reformat and plot the max change in demand data
# Event
dmd_dat.m <- melt(dmd_dat, id.vars="group")
plt_dmd <- ggplot(dmd_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Max. Decrease in Event Demand (W/sf)") + xlab("Candidate Strategy")
plt_dmd_fin <- plt_dmd + coord_flip()
# Precool
dmd_dat_pc.m <- melt(dmd_dat_pc, id.vars="group")
plt_dmd_pc <- ggplot(dmd_dat_pc.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Max. Increase in Pre-Cooling Demand (W/sf)") + xlab("Candidate Strategy")
plt_dmd_pc_fin <- plt_dmd_pc + coord_flip()

# Reformat and plot the total change in economic benefit data
# Overall
cost_dat.m <- melt(cost_dat, id.vars="group")
plt_cost <- ggplot(cost_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Total Economic Benefit (100$)") + xlab("Candidate Strategy")
plt_cost_fin <- plt_cost + coord_flip()
# Precool
cost_dat_pc.m <- melt(cost_dat_pc, id.vars="group")
plt_cost_pc <- ggplot(cost_dat_pc.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Total Economic Loss, Pre-Cooling (100$)") + xlab("Candidate Strategy")
plt_cost_pc_fin <- plt_cost_pc + coord_flip()

# Reformat and plot the max change in temperature data
# Event
tmp_dat.m <- melt(temp_dat, id.vars="group")
plt_tmp <- ggplot(tmp_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Max. Increase in Event Temp. (ºF)") + xlab("Candidate Strategy")
plt_tmp_fin <- plt_tmp + coord_flip()
# # Precool
# tmp_dat_pc.m <- melt(temp_dat_pc, id.vars="group")
# plt_tmp_pc <- ggplot(tmp_dat_pc.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Max. Decrease in Pre-Cooling Temp. (ºF)") + xlab("Candidate Strategy")
# plt_tmp_pc_fin <- plt_tmp_pc + coord_flip()

# Reformat and plot the max change in lighting data
lgt_dat.m <- melt(lgt_dat, id.vars="group")
plt_lgt <- ggplot(lgt_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + ylab("Max. Decrease in Event Lighting (frac)") + xlab("Candidate Strategy")
plt_lgt_fin <- plt_lgt + coord_flip()

# Reformat and plot the selection probability data
chc_dat.m <- melt(chc_dat, id.vars="group")
# Initialize a dataframe for storing overall frequencies of selection
chc_dat_pts <- data.frame(
	xname = dat_meas,
	ypos = chc_dat_pts/100)
# Set threshold for measure selection
index_thres = which(grepl("(D)", dat_meas, fixed=TRUE))
# Check for default option; if there is none, selection threshold is 1/total N measures
if (length(index_thres) != 0){
	threshold = chc_dat_pts$`ypos`[index_thres]
}else{
	threshold = (1 / length(dat_meas))	
}
plt_chc <- ggplot(chc_dat.m, aes(x=reorder(group, value, FUN=median), y=value)) + geom_boxplot() + 
	geom_point(data=chc_dat_pts, mapping = aes(x = xname, y = ypos), color = "green") +
	ylab("Selection Probability (frac)") + xlab("Candidate Strategy")
plt_chc_fin <- plt_chc + coord_flip() + geom_hline(yintercept = threshold, color="red", size=1)

# Organize the plots on one page
grid.arrange(plt_dmd_fin , plt_dmd_pc_fin, plt_cost_fin, plt_cost_pc_fin, plt_tmp_fin, plt_lgt_fin, plt_chc_fin, nrow = 2)
# grid.arrange(plt_cost_fin, plt_tmp_fin, plt_lgt_fin, plt_chc_fin, nrow = 1)
dev.off()